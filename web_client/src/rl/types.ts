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
}
