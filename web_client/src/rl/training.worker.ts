/**
 * Training Worker - runs neural network training in a separate thread
 * Handles all expensive operations: replay buffer, backprop, weight updates
 */

import { GameEngine, DefaultObservationConfig, DefaultRewardConfig, type ObservationConfig, type RewardConfig } from '../game'
import { NeuralNetwork, createDQNNetwork } from './NeuralNetwork'
import { ReplayBuffer, type Transition } from './ReplayBuffer'
import type { DQNConfig } from './DQNAgent'
import type { TrainingMetrics } from './types'

// Message types sent TO worker
type WorkerMessage =
  | { type: 'init'; config: DQNConfig }
  | { type: 'experience'; transition: Transition }
  | { type: 'requestWeights' }
  | { type: 'setWeights'; data: { weights: number[][][]; biases: number[][] } }
  | { type: 'setEpsilon'; value: number }
  | { type: 'setAutoDecay'; enabled: boolean }
  | { type: 'setLearningRate'; value: number }
  | { type: 'setGamma'; value: number }
  | { type: 'setRewardConfig'; config: Partial<RewardConfig> }
  | { type: 'startFast'; rewardConfig?: Partial<RewardConfig>; observationConfig?: Partial<ObservationConfig> }
  | { type: 'stopFast' }
  | { type: 'reset' }

// Message types sent FROM worker
type WorkerResponse =
  | { type: 'weights'; data: { weights: number[][][]; biases: number[][] }; steps: number; loss: number }
  | { type: 'ready' }
  | { type: 'fastMetrics'; metrics: TrainingMetrics }
  | { type: 'error'; message: string }

// Worker state
let policyNetwork: NeuralNetwork | null = null
let targetNetwork: NeuralNetwork | null = null
let replayBuffer: ReplayBuffer | null = null
let config: DQNConfig | null = null
let rewardConfig: RewardConfig = DefaultRewardConfig
let observationConfig: ObservationConfig = DefaultObservationConfig
let steps: number = 0
let lastLoss: number = 0
let weightUpdateCounter: number = 0
let experienceCount: number = 0

// Fast-mode environment state
let fastEngine: GameEngine | null = null
let fastModeRunning: boolean = false
let fastEpisode: number = 0
let fastEpisodeReward: number = 0
let fastEpisodeLength: number = 0
let fastRecentRewards: number[] = []
let fastRecentLengths: number[] = []
let fastLastMetricsTime: number = 0
let fastStepsSinceLastMetric: number = 0
let fastStepsPerSecond: number = 0
let fastTotalSteps: number = 0

// Training frequency controls
const TRAIN_FREQ = 4          // Train every 4 experiences (closer to Python optimize_every=1)
const WEIGHT_UPDATE_FREQ = 400 // Send weights every N training steps
const FAST_BATCH_STEPS = 512   // Steps to process per fast-mode chunk
const METRICS_WINDOW = 50

// Epsilon tracking (for logging - actual epsilon managed on main thread)
let epsilon: number = 1.0
let autoDecayEnabled: boolean = true
let decayStartEpsilon: number = 1.0
let decayStartStep: number = 0

/**
 * Train on a batch from replay buffer
 */
function trainOnBatch(): void {
  if (!policyNetwork || !targetNetwork || !replayBuffer || !config) return

  if (!replayBuffer.canSample(config.batchSize)) {
    return
  }

  const batch = replayBuffer.sample(config.batchSize)
  const states: number[][] = []
  const actions: number[] = []
  const targets: number[] = []

  for (const t of batch) {
    states.push(t.state)
    actions.push(t.action)

    // Compute target Q-value using target network
    const nextQValues = targetNetwork.predict(t.nextState)
    const maxNextQ = Math.max(...nextQValues)
    const target = t.reward + (t.done ? 0 : config.gamma * maxNextQ)
    targets.push(target)
  }

  // Train policy network
  lastLoss = policyNetwork.trainBatch(states, actions, targets)
  steps++
  weightUpdateCounter++

  // Update target network periodically
  if (steps % config.targetUpdateFreq === 0) {
    targetNetwork.copyWeightsFrom(policyNetwork)
  }

  // Send weights to main thread periodically
  if (weightUpdateCounter >= WEIGHT_UPDATE_FREQ) {
    weightUpdateCounter = 0
    const weights = policyNetwork.toJSON()
    self.postMessage({
      type: 'weights',
      data: weights,
      steps,
      loss: lastLoss,
    } as WorkerResponse)
  }
}

/**
 * Update epsilon decay (for logging purposes)
 */
function updateEpsilon(): void {
  if (!config || !autoDecayEnabled) return

  const stepsSinceDecayStart = steps - decayStartStep
  const frac = Math.min(1.0, stepsSinceDecayStart / config.epsilonDecaySteps)
  epsilon =
    decayStartEpsilon + frac * (config.epsilonEnd - decayStartEpsilon)
}

function emitFastMetrics(): void {
  if (!config || !replayBuffer) return

  const avgReward =
    fastRecentRewards.length > 0
      ? fastRecentRewards.reduce((a, b) => a + b, 0) / fastRecentRewards.length
      : 0

  const avgLength =
    fastRecentLengths.length > 0
      ? fastRecentLengths.reduce((a, b) => a + b, 0) / fastRecentLengths.length
      : 0

  const metrics: TrainingMetrics = {
    episode: fastEpisode,
    episodeReward: fastEpisodeReward,
    episodeLength: fastEpisodeLength,
    avgReward,
    avgLength,
    epsilon,
    loss: lastLoss,
    bufferSize: replayBuffer.size(),
    stepsPerSecond: fastStepsPerSecond,
    totalSteps: fastTotalSteps,
  }

  self.postMessage({ type: 'fastMetrics', metrics } as WorkerResponse)
}

function runFastModeBatch(): void {
  if (!fastModeRunning || !fastEngine || !config || !policyNetwork || !replayBuffer) {
    return
  }

  const startTime = performance.now()

  for (let i = 0; i < FAST_BATCH_STEPS && fastModeRunning; i++) {
    const state = fastEngine.getObservation()
    const qValues = policyNetwork.predict(state)
    const action = Math.random() < epsilon
      ? Math.floor(Math.random() * config.actionDim)
      : qValues[0] > qValues[1] ? 0 : 1

    const result = fastEngine.step(action as 0 | 1)

    replayBuffer.add({
      state,
      action,
      reward: result.reward,
      nextState: result.observation,
      done: result.done,
    })
    experienceCount++
    fastTotalSteps++
    fastStepsSinceLastMetric++

    if (experienceCount % TRAIN_FREQ === 0 && replayBuffer.canSample(config.batchSize)) {
      trainOnBatch()
      updateEpsilon()
    }

    fastEpisodeReward += result.reward
    fastEpisodeLength++

    if (result.done) {
      fastEpisode++
      fastRecentRewards.push(fastEpisodeReward)
      fastRecentLengths.push(fastEpisodeLength)
      if (fastRecentRewards.length > METRICS_WINDOW) {
        fastRecentRewards.shift()
        fastRecentLengths.shift()
      }

      fastEngine.reset()
      fastEpisodeReward = 0
      fastEpisodeLength = 0
    }

    // Exit early if the batch already consumed a long chunk of time
    if (performance.now() - startTime > 25) {
      break
    }
  }

  const now = performance.now()
  if (now - fastLastMetricsTime >= 500) {
    fastStepsPerSecond = (fastStepsSinceLastMetric / (now - fastLastMetricsTime)) * 1000
    fastStepsSinceLastMetric = 0
    fastLastMetricsTime = now
    emitFastMetrics()
  }

  if (fastModeRunning) {
    setTimeout(runFastModeBatch, 0)
  }
}

function startFastMode(
  rewardOverrides?: Partial<RewardConfig>,
  observationOverrides?: Partial<ObservationConfig>
): void {
  if (!config) return

  rewardConfig = { ...rewardConfig, ...rewardOverrides }
  observationConfig = { ...observationConfig, ...observationOverrides }
  fastEngine = new GameEngine(rewardConfig, observationConfig)
  fastEngine.reset()

  fastModeRunning = true
  fastEpisode = 0
  fastEpisodeReward = 0
  fastEpisodeLength = 0
  fastRecentRewards = []
  fastRecentLengths = []
  fastLastMetricsTime = performance.now()
  fastStepsSinceLastMetric = 0
  fastStepsPerSecond = 0
  fastTotalSteps = 0

  runFastModeBatch()
}

function stopFastMode(): void {
  fastModeRunning = false
}

/**
 * Handle messages from main thread
 */
self.onmessage = (e: MessageEvent<WorkerMessage>) => {
  try {
    const message = e.data

    switch (message.type) {
      case 'init': {
        config = message.config
        steps = 0
        weightUpdateCounter = 0
        epsilon = config.epsilonStart
        decayStartEpsilon = config.epsilonStart
        decayStartStep = 0
        rewardConfig = DefaultRewardConfig
        observationConfig = DefaultObservationConfig

        // Create networks
        policyNetwork = createDQNNetwork(
          config.inputDim,
          config.hiddenLayers,
          config.actionDim,
          config.learningRate
        )

        targetNetwork = createDQNNetwork(
          config.inputDim,
          config.hiddenLayers,
          config.actionDim,
          config.learningRate
        )

        // Copy weights to target
        targetNetwork.copyWeightsFrom(policyNetwork)

        // Create replay buffer
        replayBuffer = new ReplayBuffer(config.bufferSize)

        // Send initial weights
        const initialWeights = policyNetwork.toJSON()
        self.postMessage({
          type: 'weights',
          data: initialWeights,
          steps: 0,
          loss: 0,
        } as WorkerResponse)

        self.postMessage({ type: 'ready' } as WorkerResponse)
        break
      }

      case 'experience': {
        if (!replayBuffer) {
          console.warn('[Worker] Replay buffer not initialized')
          break
        }

        replayBuffer.add(message.transition)
        experienceCount++

        // Train every TRAIN_FREQ experiences (not every one - reduces CPU load)
        if (experienceCount % TRAIN_FREQ === 0 && replayBuffer.canSample(config?.batchSize || 16)) {
          trainOnBatch()
          updateEpsilon()
        }
        break
      }

      case 'requestWeights': {
        if (!policyNetwork) {
          console.warn('[Worker] Policy network not initialized')
          break
        }

        const weights = policyNetwork.toJSON()
        self.postMessage({
          type: 'weights',
          data: weights,
          steps,
          loss: lastLoss,
        } as WorkerResponse)
        break
      }

      case 'setWeights': {
        if (!policyNetwork || !targetNetwork) {
          console.warn('[Worker] Cannot set weights - networks not initialized')
          break
        }

        // Load weights into both policy and target networks
        policyNetwork.loadJSON(message.data)
        targetNetwork.loadJSON(message.data)

        // Reset counters so future training/epsilon decay behaves nicely
        weightUpdateCounter = 0
        lastLoss = 0
        break
      }

      case 'setEpsilon': {
        epsilon = Math.max(0, Math.min(1, message.value))
        break
      }

      case 'setAutoDecay': {
        const wasAutoDecay = autoDecayEnabled
        autoDecayEnabled = message.enabled
        if (message.enabled && !wasAutoDecay) {
          decayStartEpsilon = epsilon
          decayStartStep = steps
        }
        break
      }

      case 'setLearningRate': {
        if (policyNetwork) {
          policyNetwork.setLearningRate(message.value)
        }
        if (config) {
          config.learningRate = message.value
        }
        break
      }

      case 'setGamma': {
        if (config) {
          config.gamma = Math.max(0, Math.min(1, message.value))
        }
        break
      }

      case 'setRewardConfig': {
        rewardConfig = { ...rewardConfig, ...message.config }
        if (fastEngine) {
          fastEngine.setRewardConfig(message.config)
        }
        break
      }

      case 'startFast': {
        if (!policyNetwork || !replayBuffer) {
          console.warn('[Worker] Cannot start fast mode without initialization')
          break
        }
        startFastMode(message.rewardConfig, message.observationConfig)
        break
      }

      case 'stopFast': {
        stopFastMode()
        break
      }

      case 'reset': {
        stopFastMode()
        steps = 0
        weightUpdateCounter = 0
        experienceCount = 0
        epsilon = config?.epsilonStart || 1.0
        decayStartEpsilon = config?.epsilonStart || 1.0
        decayStartStep = 0
        lastLoss = 0

        if (replayBuffer) {
          replayBuffer.clear()
        }

        // Reinitialize networks
        if (config) {
          policyNetwork = createDQNNetwork(
            config.inputDim,
            config.hiddenLayers,
            config.actionDim,
            config.learningRate
          )

          targetNetwork = createDQNNetwork(
            config.inputDim,
            config.hiddenLayers,
            config.actionDim,
            config.learningRate
          )

          targetNetwork.copyWeightsFrom(policyNetwork)

          // Send reset weights
          const resetWeights = policyNetwork.toJSON()
          self.postMessage({
            type: 'weights',
            data: resetWeights,
            steps: 0,
            loss: 0,
          } as WorkerResponse)
        }
        break
      }

      default: {
        console.warn('[Worker] Unknown message type:', (message as any).type)
      }
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
    } as WorkerResponse)
  }
}

// Export types for main thread
export type { WorkerMessage, WorkerResponse }

