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
  | { type: 'setLRScheduler'; enabled: boolean }
  | { type: 'startFast'; rewardConfig?: Partial<RewardConfig>; observationConfig?: Partial<ObservationConfig>; startingEpisode?: number; startingTotalSteps?: number }
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
let experienceCount: number = 0

// Fast-mode environment state
let fastEngine: GameEngine | null = null
let fastModeRunning: boolean = false
let fastEpisode: number = 0
let fastEpisodeReward: number = 0
let fastEpisodeLength: number = 0
let fastLastCompletedReward: number = 0  // Track last completed episode's reward
let fastLastCompletedLength: number = 0  // Track last completed episode's length
let fastRecentRewards: number[] = []
let fastRecentLengths: number[] = []
let fastLastMetricsTime: number = 0
let fastStepsSinceLastMetric: number = 0
let fastStepsPerSecond: number = 0
let fastTotalSteps: number = 0

// Training frequency controls
const TRAIN_FREQ = 16         // Train every 16 experiences (faster throughput)
const FAST_BATCH_STEPS = 1024  // Steps to process per fast-mode chunk (doubled for speed)
const METRICS_WINDOW = 50
const WARMUP_SIZE = 50000     // Minimum buffer size before training starts (100% of buffer)

// Epsilon tracking (for logging - actual epsilon managed on main thread)
let epsilon: number = 1.0
let autoDecayEnabled: boolean = true
let decayStartEpsilon: number = 1.0
let decayStartStep: number = 0

// Learning rate scheduler state
let lrSchedulerEnabled: boolean = false
let currentLearningRate: number = 0.0005
let lrSchedulerBestAvgReward: number = -Infinity
let lrSchedulerPatienceCounter: number = 0
const LR_SCHEDULER_PATIENCE = 50  // Episodes without improvement before reducing LR
const LR_SCHEDULER_FACTOR = 0.5   // Multiply LR by this when reducing
const LR_SCHEDULER_MIN = 0.00001  // Minimum learning rate

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

  // Update target network periodically
  if (steps % config.targetUpdateFreq === 0) {
    targetNetwork.copyWeightsFrom(policyNetwork)
  }

  // Note: We no longer send weights periodically during fast training.
  // Weights are only synced when:
  // 1. Exiting fast mode (via requestWeights)
  // 2. Entering eval mode
  // 3. User saves a checkpoint
  // This significantly reduces overhead during fast training.
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

function isInWarmup(): boolean {
  return replayBuffer ? replayBuffer.size() < WARMUP_SIZE : true
}

/**
 * Learning rate scheduler - reduces LR when avg reward plateaus
 */
function updateLRScheduler(avgReward: number): void {
  if (!lrSchedulerEnabled || !policyNetwork) return
  
  if (avgReward > lrSchedulerBestAvgReward) {
    // Improvement - reset patience
    lrSchedulerBestAvgReward = avgReward
    lrSchedulerPatienceCounter = 0
  } else {
    // No improvement - increment patience
    lrSchedulerPatienceCounter++
    
    if (lrSchedulerPatienceCounter >= LR_SCHEDULER_PATIENCE) {
      // Reduce learning rate
      const newLR = Math.max(LR_SCHEDULER_MIN, currentLearningRate * LR_SCHEDULER_FACTOR)
      if (newLR < currentLearningRate) {
        currentLearningRate = newLR
        policyNetwork.setLearningRate(newLR)
        console.log(`[Worker] LR Scheduler: Reduced learning rate to ${newLR.toExponential(2)}`)
      }
      lrSchedulerPatienceCounter = 0
    }
  }
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
    episodeReward: fastLastCompletedReward,  // Report last completed episode, not in-progress
    episodeLength: fastLastCompletedLength,
    avgReward,
    avgLength,
    epsilon,
    loss: lastLoss,
    bufferSize: replayBuffer.size(),
    stepsPerSecond: fastStepsPerSecond,
    totalSteps: fastTotalSteps,
    isWarmup: isInWarmup(),
    learningRate: currentLearningRate,
  }

  // Update LR scheduler based on avg reward
  updateLRScheduler(avgReward)

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

    // Only train after warmup phase (buffer has enough samples)
    if (!isInWarmup() && experienceCount % TRAIN_FREQ === 0 && replayBuffer.canSample(config.batchSize)) {
      trainOnBatch()
      updateEpsilon()
    }

    fastEpisodeReward += result.reward
    fastEpisodeLength++

    if (result.done) {
      fastEpisode++
      // Store completed episode stats before resetting
      fastLastCompletedReward = fastEpisodeReward
      fastLastCompletedLength = fastEpisodeLength
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
    // Allow up to 50ms per batch for better throughput
    if (performance.now() - startTime > 50) {
      break
    }
  }

  const now = performance.now()
  // Emit metrics more frequently during warmup (every 100ms) so user can see progress
  // After warmup, emit every 500ms to reduce overhead
  const metricsInterval = isInWarmup() ? 100 : 500
  if (now - fastLastMetricsTime >= metricsInterval) {
    fastStepsPerSecond = (fastStepsSinceLastMetric / (now - fastLastMetricsTime)) * 1000
    fastStepsSinceLastMetric = 0
    fastLastMetricsTime = now
    emitFastMetrics()
  }

  if (fastModeRunning) {
    // Use setImmediate-like pattern for better throughput
    setTimeout(runFastModeBatch, 0)
  }
}

function startFastMode(
  rewardOverrides?: Partial<RewardConfig>,
  observationOverrides?: Partial<ObservationConfig>,
  startingEpisode: number = 0,
  startingTotalSteps: number = 0
): void {
  if (!config) return

  rewardConfig = { ...rewardConfig, ...rewardOverrides }
  observationConfig = { ...observationConfig, ...observationOverrides }
  fastEngine = new GameEngine(rewardConfig, observationConfig)
  fastEngine.reset()

  fastModeRunning = true
  fastEpisode = startingEpisode  // Start from the passed episode count
  fastEpisodeReward = 0
  fastEpisodeLength = 0
  fastLastCompletedReward = 0
  fastLastCompletedLength = 0
  fastRecentRewards = []
  fastRecentLengths = []
  fastLastMetricsTime = performance.now()
  fastStepsSinceLastMetric = 0
  fastStepsPerSecond = 0
  fastTotalSteps = startingTotalSteps  // Start from the passed step count

  // Emit metrics immediately so UI shows warmup state right away
  emitFastMetrics()
  
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
        epsilon = config.epsilonStart
        decayStartEpsilon = config.epsilonStart
        decayStartStep = 0
        currentLearningRate = config.learningRate
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

        // Only train after warmup phase (buffer has enough samples)
        if (!isInWarmup() && experienceCount % TRAIN_FREQ === 0 && replayBuffer.canSample(config?.batchSize || 16)) {
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
        currentLearningRate = message.value
        if (policyNetwork) {
          policyNetwork.setLearningRate(message.value)
        }
        if (config) {
          config.learningRate = message.value
        }
        break
      }

      case 'setLRScheduler': {
        lrSchedulerEnabled = message.enabled
        if (message.enabled) {
          // Reset scheduler state when enabling
          lrSchedulerBestAvgReward = -Infinity
          lrSchedulerPatienceCounter = 0
        }
        console.log(`[Worker] LR Scheduler: ${message.enabled ? 'enabled' : 'disabled'}`)
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
        startFastMode(message.rewardConfig, message.observationConfig, message.startingEpisode, message.startingTotalSteps)
        break
      }

      case 'stopFast': {
        stopFastMode()
        break
      }

      case 'reset': {
        stopFastMode()
        steps = 0
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

