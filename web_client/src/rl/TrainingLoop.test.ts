import { describe, expect, it, vi, beforeEach } from 'vitest'
import { TrainingLoop } from './TrainingLoop'
import type { TrainingMetrics } from './types'
import type { RewardConfig, StepResult } from '@/game'

class StubEngine {
  resetCalls = 0
  stepCalls = 0
  rewardConfig: Partial<RewardConfig> = {}

  reset(): number[] {
    this.resetCalls += 1
    return [0, 0, 0, 0, 0, 0]
  }

  step(): StepResult {
    this.stepCalls += 1
    return {
      observation: [1, 1, 1, 1, 1, 1],
      reward: 1,
      done: false,
      info: { episode: 0, score: 0, steps: this.stepCalls },
    }
  }

  setRewardConfig(config: Partial<RewardConfig>): void {
    this.rewardConfig = { ...this.rewardConfig, ...config }
  }
}

class StubAgent {
  startCalls = 0
  stopCalls = 0
  weightRequests = 0
  weightSyncs = 0
  fastMetricsCallback: ((metrics: TrainingMetrics) => void) | null = null
  networkVizCalls = 0

  act(): 0 | 1 {
    return 0
  }

  remember(): void {}

  replay(): void {}

  getSteps(): number {
    return 0
  }

  getEpsilon(): number {
    return 0.5
  }

  getLastLoss(): number {
    return 0
  }

  getBufferSize(): number {
    return 0
  }

  getLastQValues(): number[] {
    return [0, 0]
  }

  getNetworkVisualization() {
    this.networkVizCalls += 1
    return { activations: [], qValues: [0, 0], selectedAction: 0 }
  }

  setRewardConfig(): void {}

  isUsingWorker(): boolean {
    return true
  }

  syncWeightsToWorker(): void {
    this.weightSyncs += 1
  }

  startFastTraining(): void {
    this.startCalls += 1
  }

  stopFastTraining(): void {
    this.stopCalls += 1
  }

  requestWeights(): void {
    this.weightRequests += 1
  }

  onFastMetrics(cb: (metrics: TrainingMetrics) => void): void {
    this.fastMetricsCallback = cb
  }

  emitFastMetrics(metrics: TrainingMetrics): void {
    this.fastMetricsCallback?.(metrics)
  }
}

describe('TrainingLoop fast-mode control', () => {
  let engine: StubEngine
  let agent: StubAgent

  beforeEach(() => {
    engine = new StubEngine()
    agent = new StubAgent()
  })

  it('forwards worker metrics and skips main-thread stepping while fast mode is active', () => {
    const onStep = vi.fn()
    const loop = new TrainingLoop(engine as any, {}, { onStep }, () => agent as any)

    loop.start()
    expect(engine.resetCalls).toBe(1)

    loop.setFastMode(true)
    expect(agent.startCalls).toBe(1)

    // Step should be ignored when fast mode is running
    const result = loop.step()
    expect(result).toBeNull()
    expect(engine.stepCalls).toBe(0)

    const metrics: TrainingMetrics = {
      episode: 3,
      episodeReward: 12,
      episodeLength: 55,
      avgReward: 8,
      avgLength: 40,
      epsilon: 0.3,
      loss: 0.01,
      bufferSize: 1000,
      stepsPerSecond: 2500,
      totalSteps: 1234,
    }

    agent.emitFastMetrics(metrics)

    expect(onStep).toHaveBeenCalledWith(metrics, [0, 0])
    expect(loop.getMetrics()).toEqual(metrics)
  })

  it('resets episode stats and resumes main-thread stepping after fast mode stops', () => {
    const loop = new TrainingLoop(engine as any, {}, {}, () => agent as any)

    loop.start()
    loop.setFastMode(true)

    expect(agent.weightSyncs).toBe(1)

    agent.emitFastMetrics({
      episode: 5,
      episodeReward: 20,
      episodeLength: 80,
      avgReward: 10,
      avgLength: 60,
      epsilon: 0.2,
      loss: 0.05,
      bufferSize: 2000,
      stepsPerSecond: 3000,
      totalSteps: 2000,
    })

    loop.setFastMode(false)

    expect(agent.stopCalls).toBe(1)
    expect(agent.weightRequests).toBe(1)
    expect(engine.resetCalls).toBe(2)

    const metrics = loop.getMetrics()
    expect(metrics.episodeReward).toBe(0)
    expect(metrics.episodeLength).toBe(0)

    loop.step()
    expect(engine.stepCalls).toBe(1)
  })

  it('avoids invoking agent visualization while fast mode is running', () => {
    const loop = new TrainingLoop(engine as any, {}, {}, () => agent as any)

    loop.start()
    loop.setFastMode(true)

    const viz = loop.getNetworkVisualization()

    expect(viz).toEqual({ activations: [], qValues: [0, 0], selectedAction: 0 })
    expect(agent.networkVizCalls).toBe(0)
  })
})

