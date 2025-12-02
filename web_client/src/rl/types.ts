export interface TrainingMetrics {
  episode: number
  episodeReward: number
  episodeLength: number
  avgReward: number
  avgLength: number
  epsilon: number
  loss: number
  bufferSize: number
  stepsPerSecond: number
  totalSteps: number
  isWarmup: boolean  // True during warmup phase (collecting experience before training)
  learningRate: number  // Current learning rate (may change with scheduler)
}
