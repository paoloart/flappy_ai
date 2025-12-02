/**
 * Worker-based DQN Agent - lightweight main-thread agent that delegates training to a Web Worker
 * Falls back to main-thread training if Web Workers are not available
 */

import type { RewardConfig } from '@/game'
import { NeuralNetwork, createDQNNetwork } from './NeuralNetwork'
import { ReplayBuffer, type Transition } from './ReplayBuffer'
import type { TrainingMetrics } from './types'

export interface DQNConfig {
  // Network architecture
  inputDim: number
  hiddenLayers: number[]
  actionDim: number

  // Training hyperparameters
  learningRate: number
  gamma: number
  batchSize: number
  bufferSize: number

  // Epsilon-greedy
  epsilonStart: number
  epsilonEnd: number
  epsilonDecaySteps: number

  // Target network
  targetUpdateFreq: number
}

export const DefaultDQNConfig: DQNConfig = {
  inputDim: 6, // 6 base features (no gap velocities - pipes are static)
  hiddenLayers: [64, 64],
  actionDim: 2,

  learningRate: 0.001,
  gamma: 0.99,
  batchSize: 32,
  bufferSize: 50000,

  epsilonStart: 0.5,
  epsilonEnd: 0.05,
  epsilonDecaySteps: 150000,

  targetUpdateFreq: 200,
}

export class WorkerDQNAgent {
  private config: DQNConfig
  private worker: Worker | null = null
  private inferenceNetwork: NeuralNetwork
  private workerSupported: boolean

  // Fallback training state (used when worker fails)
  private fallbackTargetNetwork: NeuralNetwork | null = null
  private fallbackBuffer: ReplayBuffer | null = null
  private trainingSteps: number = 0

  // Training state (tracked on main thread)
  private steps: number = 0
  private epsilon: number
  private autoDecayEnabled: boolean = true
  private decayStartEpsilon: number
  private decayStartStep: number = 0

  // Metrics from worker
  private lastLoss: number = 0
  private bufferSize: number = 0
  private lastQValues: number[] = [0, 0]
  private lastWorkerMetrics: TrainingMetrics | null = null
  private fastMetricsCallback?: (metrics: TrainingMetrics) => void

  // Worker ready state
  private workerReady: boolean = false
  private pendingExperiences: Transition[] = []

  constructor(config: Partial<DQNConfig> = {}) {
    this.config = { ...DefaultDQNConfig, ...config }
    this.epsilon = this.config.epsilonStart
    this.decayStartEpsilon = this.config.epsilonStart
    this.decayStartStep = 0

    // Check for Web Worker support
    this.workerSupported = typeof Worker !== 'undefined'

    // Create local inference network
    this.inferenceNetwork = createDQNNetwork(
      this.config.inputDim,
      this.config.hiddenLayers,
      this.config.actionDim,
      this.config.learningRate
    )

    if (this.workerSupported) {
      this.initWorker()
    } else {
      console.warn('[WorkerDQNAgent] Web Workers not supported')
      this.initFallback()
    }

    console.log('[WorkerDQNAgent] Created with Web Worker:', this.isUsingWorker())
    console.log('[WorkerDQNAgent] Layer sizes:', this.inferenceNetwork.getLayerSizes())
  }

  /**
   * Initialize fallback training (main thread)
   */
  private initFallback(): void {
    console.log('[WorkerDQNAgent] Initializing fallback training on main thread')
    
    // Create target network
    this.fallbackTargetNetwork = createDQNNetwork(
      this.config.inputDim,
      this.config.hiddenLayers,
      this.config.actionDim,
      this.config.learningRate
    )
    this.fallbackTargetNetwork.copyWeightsFrom(this.inferenceNetwork)

    // Create replay buffer
    this.fallbackBuffer = new ReplayBuffer(this.config.bufferSize)
  }

  /**
   * Initialize the Web Worker
   */
  private initWorker(): void {
    try {
      // Vite handles worker bundling with this syntax
      this.worker = new Worker(
        new URL('./training.worker.ts', import.meta.url),
        { type: 'module' }
      )

      console.log('[WorkerDQNAgent] Worker created successfully')

      this.worker.onmessage = (e) => {
        const message = e.data

        switch (message.type) {
          case 'ready':
            console.log('[WorkerDQNAgent] Worker ready')
            this.workerReady = true
            // Send any pending experiences
            for (const exp of this.pendingExperiences) {
              this.worker?.postMessage({ type: 'experience', transition: exp })
            }
            this.pendingExperiences = []
            break

          case 'weights':
            // Update local inference network with weights from worker
            this.inferenceNetwork.loadJSON(message.data)
            this.lastLoss = message.loss || 0
            // Don't overwrite steps - main thread tracks steps for epsilon decay
            break

          case 'fastMetrics':
            this.lastLoss = message.metrics.loss
            this.bufferSize = message.metrics.bufferSize
            this.lastWorkerMetrics = message.metrics
            this.fastMetricsCallback?.(message.metrics)
            break

          case 'error':
            console.error('[WorkerDQNAgent] Worker error:', message.message)
            break
        }
      }

      this.worker.onerror = (error) => {
        console.error('[WorkerDQNAgent] Worker error:', error)
        // If worker fails after creation, set up fallback
        if (!this.fallbackBuffer) {
          this.initFallback()
        }
        this.workerReady = false
      }

      // Initialize worker with config (JSON serialize to ensure clonability)
      const serializableConfig = JSON.parse(JSON.stringify(this.config))
      this.worker.postMessage({ type: 'init', config: serializableConfig })
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error)
      console.warn('[WorkerDQNAgent] Failed to create worker:', errMsg)
      console.warn('[WorkerDQNAgent] Falling back to main-thread training')
      this.workerSupported = false
      this.worker = null
      this.initFallback()
    }
  }

  /**
   * Select action using epsilon-greedy policy
   * Runs on main thread for immediate response
   */
  act(state: number[], training: boolean = true): number {
    if (training) {
      this.steps++
      if (this.autoDecayEnabled) {
        this.updateEpsilon()
      }
    }

    // Always compute Q-values for visualization (even when exploring)
    this.lastQValues = this.inferenceNetwork.predict(state)

    // Epsilon-greedy exploration
    if (training && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.config.actionDim)
    }

    // Greedy action
    return this.lastQValues[0] > this.lastQValues[1] ? 0 : 1
  }

  /**
   * Store transition - sends to worker or fallback buffer
   */
  remember(transition: Transition): void {
    this.bufferSize = Math.min(this.bufferSize + 1, this.config.bufferSize)

    if (this.worker && this.workerReady) {
      // Worker mode: send to worker (ensure serializable)
      try {
        const serializableTransition = {
          state: [...transition.state],
          action: transition.action,
          reward: transition.reward,
          nextState: [...transition.nextState],
          done: transition.done,
        }
        this.worker.postMessage({ type: 'experience', transition: serializableTransition })
      } catch (error) {
        // If worker communication fails, fall back to local training
        console.warn('[WorkerDQNAgent] Failed to send to worker, using fallback')
        if (!this.fallbackBuffer) this.initFallback()
        this.fallbackBuffer?.add(transition)
      }
    } else if (this.worker) {
      // Worker exists but not ready: queue
      this.pendingExperiences.push(transition)
      if (this.pendingExperiences.length > this.config.bufferSize) {
        this.pendingExperiences.shift()
      }
    } else if (this.fallbackBuffer) {
      // Fallback mode: add to local buffer
      this.fallbackBuffer.add(transition)
    }
  }

  /**
   * Train the network - only does work in fallback mode
   * Worker mode trains automatically
   */
  replay(): boolean {
    // Worker mode: training happens in worker automatically
    if (this.worker && this.workerReady) {
      return true
    }

    // Fallback mode: train on main thread
    if (!this.fallbackBuffer || !this.fallbackTargetNetwork) {
      return false
    }

    if (!this.fallbackBuffer.canSample(this.config.batchSize)) {
      return false
    }

    const batch = this.fallbackBuffer.sample(this.config.batchSize)
    let totalLoss = 0

    for (const t of batch) {
      // Compute target Q-value
      const nextQValues = this.fallbackTargetNetwork.predict(t.nextState)
      const maxNextQ = Math.max(...nextQValues)
      const target = t.reward + (t.done ? 0 : this.config.gamma * maxNextQ)

      // Train
      const loss = this.inferenceNetwork.trainStep(t.state, t.action, target)
      totalLoss += loss
    }

    this.lastLoss = totalLoss / batch.length
    this.trainingSteps++

    // Update target network periodically
    if (this.trainingSteps % this.config.targetUpdateFreq === 0) {
      this.fallbackTargetNetwork.copyWeightsFrom(this.inferenceNetwork)
    }

    return true
  }

  /**
   * Update epsilon based on decay schedule
   */
  private updateEpsilon(): void {
    const stepsSinceDecayStart = this.steps - this.decayStartStep
    const frac = Math.min(1.0, stepsSinceDecayStart / this.config.epsilonDecaySteps)
    this.epsilon =
      this.decayStartEpsilon +
      frac * (this.config.epsilonEnd - this.decayStartEpsilon)
  }

  // ===== Getters and Setters =====

  getEpsilon(): number {
    return this.epsilon
  }

  setEpsilon(value: number): void {
    this.epsilon = Math.max(0, Math.min(1, value))
    this.worker?.postMessage({ type: 'setEpsilon', value: this.epsilon })
  }

  getAutoDecay(): boolean {
    return this.autoDecayEnabled
  }

  setAutoDecay(enabled: boolean): void {
    if (enabled && !this.autoDecayEnabled) {
      this.decayStartEpsilon = this.epsilon
      this.decayStartStep = this.steps
    }
    this.autoDecayEnabled = enabled
    this.worker?.postMessage({ type: 'setAutoDecay', enabled })
  }

  getEpsilonDecaySteps(): number {
    return this.config.epsilonDecaySteps
  }

  setEpsilonDecaySteps(steps: number): void {
    this.config.epsilonDecaySteps = Math.max(10000, steps)
  }

  getLearningRate(): number {
    return this.config.learningRate
  }

  setLearningRate(lr: number): void {
    this.config.learningRate = lr
    this.inferenceNetwork.setLearningRate(lr)
    this.fallbackTargetNetwork?.setLearningRate(lr)
    this.worker?.postMessage({ type: 'setLearningRate', value: lr })
  }

  getGamma(): number {
    return this.config.gamma
  }

  setGamma(value: number): void {
    this.config.gamma = Math.max(0, Math.min(1, value))
    this.worker?.postMessage({ type: 'setGamma', value: this.config.gamma })
  }

  getSteps(): number {
    return this.steps
  }

  getLastLoss(): number {
    return this.lastLoss
  }

  getLastQValues(): number[] {
    return this.lastQValues
  }

  getBufferSize(): number {
    return this.bufferSize
  }

  onFastMetrics(callback: (metrics: TrainingMetrics) => void): void {
    this.fastMetricsCallback = callback
  }

  syncWeightsToWorker(): void {
    if (this.worker && this.workerReady) {
      const weights = this.inferenceNetwork.toJSON()
      this.worker.postMessage({ type: 'setWeights', data: weights })
    }
  }

  startFastTraining(): void {
    if (this.worker && this.workerReady) {
      this.worker.postMessage({ type: 'startFast' })
    }
  }

  stopFastTraining(): void {
    if (this.worker && this.workerReady) {
      this.worker.postMessage({ type: 'stopFast' })
    }
  }

  getWorkerMetrics(): TrainingMetrics | null {
    return this.lastWorkerMetrics
  }

  setRewardConfig(config: Partial<RewardConfig>): void {
    if (this.worker && this.workerReady) {
      this.worker.postMessage({ type: 'setRewardConfig', config })
    }
  }

  /**
   * Get activations for visualization
   */
  getNetworkVisualization(state: number[]): {
    activations: number[][]
    qValues: number[]
    selectedAction: number
  } {
    const qValues = this.inferenceNetwork.predict(state)
    const activations = this.inferenceNetwork.getActivations()
    const selectedAction = qValues[0] > qValues[1] ? 0 : 1

    return {
      activations,
      qValues,
      selectedAction,
    }
  }

  /**
   * Reset training state
   */
  reset(): void {
    this.steps = 0
    this.trainingSteps = 0
    this.epsilon = this.config.epsilonStart
    this.decayStartEpsilon = this.config.epsilonStart
    this.decayStartStep = 0
    this.lastLoss = 0
    this.bufferSize = 0
    this.pendingExperiences = []

    // Reinitialize local inference network
    this.inferenceNetwork = createDQNNetwork(
      this.config.inputDim,
      this.config.hiddenLayers,
      this.config.actionDim,
      this.config.learningRate
    )

    // Reset fallback state
    if (this.fallbackBuffer) {
      this.fallbackBuffer.clear()
      this.fallbackTargetNetwork = createDQNNetwork(
        this.config.inputDim,
        this.config.hiddenLayers,
        this.config.actionDim,
        this.config.learningRate
      )
      this.fallbackTargetNetwork.copyWeightsFrom(this.inferenceNetwork)
    }

    // Reset worker
    this.worker?.postMessage({ type: 'reset' })
  }

  /**
   * Request latest weights from worker
   */
  requestWeights(): void {
    this.worker?.postMessage({ type: 'requestWeights' })
  }

  /**
   * Save model weights
   */
  save(): { weights: number[][][]; biases: number[][] } {
    return this.inferenceNetwork.toJSON()
  }

  /**
   * Load model weights
   */
  load(data: { weights: number[][][]; biases: number[][] }): void {
    this.inferenceNetwork.loadJSON(data)
    this.fallbackTargetNetwork?.loadJSON(data)

    // Also push weights to worker so policy/target networks there match
    if (this.worker && this.workerReady) {
      this.worker.postMessage({ type: 'setWeights', data })
    }
  }

  /**
   * Terminate the worker
   */
  terminate(): void {
    if (this.worker) {
      this.worker.terminate()
      this.worker = null
      this.workerReady = false
    }
  }

  /**
   * Check if using worker
   */
  isUsingWorker(): boolean {
    return this.worker !== null && this.workerReady
  }
}
