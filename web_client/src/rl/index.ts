/**
 * RL module exports
 * Uses custom NeuralNetwork with Web Worker for training
 */

export { ReplayBuffer, type Transition } from './ReplayBuffer'
export { NeuralNetwork, createDQNNetwork, type NetworkConfig } from './NeuralNetwork'

// Legacy DQNAgent (for reference/fallback)
export { DQNAgent } from './DQNAgent'

// Worker-based agent (recommended)
export { WorkerDQNAgent, DefaultDQNConfig, type DQNConfig } from './WorkerDQNAgent'

export { TrainingLoop, type TrainingCallbacks } from './TrainingLoop'
export type { TrainingMetrics } from './types'

// Network visualization type (simplified - no weights)
export interface NetworkVisualization {
  activations: number[][]
  qValues: number[]
  selectedAction: number
}
