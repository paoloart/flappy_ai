/**
 * GPU Training Worker - runs neural network training with TensorFlow.js (WebGPU/WebGL)
 * Supports multi-bird parallel simulation for maximum GPU efficiency
 */

import { GameEngine, DefaultObservationConfig, DefaultRewardConfig, type ObservationConfig, type RewardConfig } from '../game'
import { NeuralNetworkTF, createDQNNetworkTF, initTensorFlow } from './NeuralNetworkTF'
import { ReplayBuffer, type Transition } from './ReplayBuffer'
import type { TrainingMetrics, AutoEvalResult } from './types'

// ============================================================================
// Message Types
// ============================================================================

type GPUWorkerMessage =
  | { type: 'init'; config: GPUDQNConfig }
  | { type: 'setNumBirds'; value: number }
  | { type: 'setBatchSize'; value: number }
  | { type: 'experience'; transition: Transition }
  | { type: 'requestWeights' }
  | { type: 'setWeights'; data: { weights: number[][][]; biases: number[][] } }
  | { type: 'setEpsilon'; value: number }
  | { type: 'setAutoDecay'; enabled: boolean }
  | { type: 'setLearningRate'; value: number }
  | { type: 'setGamma'; value: number }
  | { type: 'setRewardConfig'; config: Partial<RewardConfig> }
  | { type: 'setAutoEval'; enabled: boolean; interval?: number; trials?: number }
  | { type: 'startFast'; startingEpisode?: number; startingTotalSteps?: number }
  | { type: 'stopFast' }
  | { type: 'reset' }
  | { type: 'dispose' }

type GPUWorkerResponse =
  | { type: 'ready'; backend: string; gpuAvailable: boolean }
  | { type: 'weights'; data: { weights: number[][][]; biases: number[][] }; steps: number; loss: number }
  | { type: 'fastMetrics'; metrics: TrainingMetrics }
  | { type: 'autoEvalResult'; result: AutoEvalResult }
  | { type: 'error'; message: string }

export interface GPUDQNConfig {
  inputDim: number
  hiddenLayers: number[]
  actionDim: number
  learningRate: number
  gamma: number
  batchSize: number
  bufferSize: number
  epsilonStart: number
  epsilonEnd: number
  epsilonDecaySteps: number
  targetUpdateFreq: number
  numBirds: number  // Number of parallel bird simulations
}

// ============================================================================
// Worker State
// ============================================================================

let policyNetwork: NeuralNetworkTF | null = null
let targetNetwork: NeuralNetworkTF | null = null
let replayBuffer: ReplayBuffer | null = null
let config: GPUDQNConfig | null = null
let rewardConfig: RewardConfig = DefaultRewardConfig
let observationConfig: ObservationConfig = DefaultObservationConfig

// Training state
let steps: number = 0  // Training step count
let lastLoss: number = 0
let experienceCount: number = 0

// Epsilon state - LINEAR decay based on TRAINING STEPS (same as CPU worker)
let epsilon: number = 1.0
let autoDecayEnabled: boolean = true
let decayStartEpsilon: number = 1.0
let decayStartTrainStep: number = 0  // Training step when decay started

// Multi-bird state
let engines: GameEngine[] = []
let numBirds: number = 1

// Fast mode state
let fastModeRunning: boolean = false
let fastEpisode: number = 0
let fastEpisodeRewards: number[] = []  // Per-bird episode rewards
let fastEpisodeLengths: number[] = []  // Per-bird episode lengths
let fastLastCompletedReward: number = 0
let fastLastCompletedLength: number = 0
let fastRecentRewards: number[] = []
let fastRecentLengths: number[] = []
let fastLastMetricsTime: number = 0
let fastStepsSinceLastMetric: number = 0
let fastStepsPerSecond: number = 0
let fastTotalSteps: number = 0
let pendingExperienceDebt: number = 0  // Tracks exp that still need training

// Auto-eval state (GPU worker)
// Di default l'auto-eval è ABILITATO, ma implementato in modo "non bloccante":
// viene eseguita in piccoli batch in parallelo al training fast-mode.
let autoEvalEnabled: boolean = true
// Esegui ogni N episodi (abbastanza alto per non disturbare troppo il training)
let autoEvalInterval: number = 5000
// Numero di episodi per auto-eval (se abilitata esplicitamente)
let autoEvalTrials: number = 20
let autoEvalRunning: boolean = false
let autoEvalCurrentTrial: number = 0
let autoEvalScores: number[] = []
// Multi-bird auto-eval engines (parallel greedy evaluation)
let autoEvalEngines: GameEngine[] = []
let autoEvalEpisodeScores: number[] = []
let lastAutoEvalEpisode: number = 0

// Constants - CRITICAL for learning
const WARMUP_SIZE = 50000  // Wait for substantial experience before training
const METRICS_WINDOW = 100  // Average over more episodes for stable metrics
const FAST_BATCH_STEPS = 128  // Steps per bird per batch
const TARGET_TRAIN_RATIO = 8  // Aim for 1 training per 8 env steps (aligned with CPU)
const WEIGHT_SYNC_INTERVAL = 500  // Sync weights to main thread every N train steps
const TARGET_UPDATE_ENV_STEPS = 10000  // Update target network every N env steps
const MAX_TRAIN_BATCHES_PER_LOOP = 512  // Hard safety cap to avoid starving UI
const TRAIN_TIME_BUDGET_MS = 25  // Max time per loop spent on backprop

// ============================================================================
// Helper Functions
// ============================================================================

function isInWarmup(): boolean {
  return replayBuffer ? replayBuffer.size() < WARMUP_SIZE : true
}

/**
 * Calculate optimal batch size based on number of birds
 * More birds = more experiences = can use larger batches for better GPU utilization
 * Larger batches are more GPU-efficient but may hurt learning stability
 */
function getOptimalBatchSize(birds: number): number {
  // Keep batch sizes moderate for stability
  // Too large batches can hurt gradient quality
  if (birds >= 2000) return 512       // Large: good GPU utilization
  if (birds >= 500) return 256        // Medium: balanced
  if (birds >= 100) return 128        // Standard: stable learning
  return 64                           // Small: conservative
}

/**
 * ===== Auto-eval helpers =====
 * GPU version esegue l'auto-valutazione in PARALLELO con più bird.
 * Usiamo più GameEngine in batch (greedy, ε=0) per sfruttare la GPU
 * e completare rapidamente, pur mantenendo 100 trial totali.
 */
const AUTO_EVAL_PARALLEL_BIRDS = 16  // Numero di bird usati durante l'auto-eval

function runAutoEval(): void {
  if (!policyNetwork || !config || autoEvalRunning) return

  autoEvalRunning = true
  autoEvalCurrentTrial = 0
  autoEvalScores = []

  // Inizializza un piccolo gruppo di bird solo per l'auto-eval (greedy)
  const evalBirds = Math.max(1, Math.min(AUTO_EVAL_PARALLEL_BIRDS, numBirds || AUTO_EVAL_PARALLEL_BIRDS))
  autoEvalEngines = []
  autoEvalEpisodeScores = []

  for (let i = 0; i < evalBirds; i++) {
    const engine = new GameEngine(rewardConfig, observationConfig)
    engine.reset()
    autoEvalEngines.push(engine)
    autoEvalEpisodeScores.push(0)
  }

  // Prima emissione per far vedere subito lo stato "AUTO-EVAL"
  emitAutoEvalProgress()

  runAutoEvalBatch()
}

/**
 * Esegue passi di auto-eval in batch, in modo non bloccante.
 * Usa predictBatch() sulla policy network per tutti i bird in parallelo.
 */
function runAutoEvalBatch(): void {
  if (!autoEvalRunning || !policyNetwork || autoEvalEngines.length === 0) {
    finishAutoEval()
    return
  }

  const MAX_EVAL_STEP_TIME_MS = 25  // Budget di tempo per batch di auto-eval
  const start = performance.now()

  while (
    performance.now() - start < MAX_EVAL_STEP_TIME_MS &&
    autoEvalCurrentTrial < autoEvalTrials
  ) {
    const n = autoEvalEngines.length
    const states: number[][] = []

    for (let i = 0; i < n; i++) {
      states.push(autoEvalEngines[i].getObservation())
    }

    // Inference greedy (ε=0) in batch
    const qValues = policyNetwork.predictBatch(states)

    for (let i = 0; i < n && autoEvalCurrentTrial < autoEvalTrials; i++) {
      const qv = qValues[i]
      const action = qv[0] > qv[1] ? 0 : 1

      const result = autoEvalEngines[i].step(action as 0 | 1)

      if (result.done) {
        // Episodio completato per questo bird
        autoEvalScores.push(result.info.score)
        autoEvalCurrentTrial++

        // Reset per nuovo episodio se dobbiamo ancora continuare
        if (autoEvalCurrentTrial < autoEvalTrials) {
          autoEvalEngines[i].reset()
        }
      }
    }
  }

  // Aggiorna le metriche di avanzamento (numero trial completati, ecc.)
  emitAutoEvalProgress()

  if (autoEvalCurrentTrial >= autoEvalTrials) {
    finishAutoEval()
  } else if (autoEvalRunning) {
    // Continua l'auto-eval nel prossimo tick per non bloccare il worker
    setTimeout(runAutoEvalBatch, 0)
  }
}

function emitAutoEvalProgress(): void {
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
    episodeReward: fastLastCompletedReward,
    episodeLength: fastLastCompletedLength,
    avgReward,
    avgLength,
    epsilon,
    loss: lastLoss,
    bufferSize: replayBuffer.size(),
    stepsPerSecond: 0,
    totalSteps: fastTotalSteps,
    isWarmup: false,
    learningRate: policyNetwork?.getLearningRate() ?? 0.0005,
    numBirds,
    batchSize: config.batchSize,
    trainSteps: trainCallCount,
    isAutoEval: true,
    autoEvalTrial: autoEvalCurrentTrial,
    autoEvalTrials: autoEvalTrials,
  }

  self.postMessage({ type: 'fastMetrics', metrics } as GPUWorkerResponse)
}

function finishAutoEval(): void {
  autoEvalRunning = false
  autoEvalEngines = []
  autoEvalEpisodeScores = []

  if (autoEvalScores.length > 0) {
    const result: AutoEvalResult = {
      avgScore: autoEvalScores.reduce((a, b) => a + b, 0) / autoEvalScores.length,
      maxScore: Math.max(...autoEvalScores),
      minScore: Math.min(...autoEvalScores),
      scores: [...autoEvalScores],
      episode: fastEpisode,
    }

    self.postMessage({ type: 'autoEvalResult', result } as GPUWorkerResponse)
    lastAutoEvalEpisode = fastEpisode
  }

  if (fastModeRunning) {
    fastLastMetricsTime = performance.now()
    runFastModeBatch()
  }
}

/**
 * Update epsilon decay - LINEAR decay based on TRAINING STEPS (same as CPU worker)
 * This is the original fork behavior: epsilon decays linearly over epsilonDecaySteps training steps.
 */
function updateEpsilon(): void {
  if (!config || !autoDecayEnabled) return
  
  const stepsSinceDecayStart = trainCallCount - decayStartTrainStep
  const frac = Math.min(1.0, stepsSinceDecayStart / config.epsilonDecaySteps)
  epsilon = decayStartEpsilon + frac * (config.epsilonEnd - decayStartEpsilon)
}

function initializeEngines(count: number): void {
  // Dispose existing engines
  engines = []
  fastEpisodeRewards = []
  fastEpisodeLengths = []
  
  // Create new engines
  for (let i = 0; i < count; i++) {
    const engine = new GameEngine(rewardConfig, observationConfig)
    engine.reset()
    engines.push(engine)
    fastEpisodeRewards.push(0)
    fastEpisodeLengths.push(0)
  }
  
  numBirds = count
  console.log(`[GPU Worker] Initialized ${count} bird engines`)
}

/**
 * Run batch inference on all birds and step all engines
 * Returns number of transitions added
 */
function stepAllBirds(): number {
  if (!policyNetwork || !config || !replayBuffer) return 0
  
  const n = engines.length
  if (n === 0) return 0
  
  // Collect states from all birds
  const states: number[][] = []
  for (let i = 0; i < n; i++) {
    states.push(engines[i].getObservation())
  }
  
  // Batch inference - this is where GPU shines!
  const qValues = policyNetwork.predictBatch(states)
  
  // Select actions (epsilon-greedy)
  const actions: number[] = []
  for (let i = 0; i < n; i++) {
    if (Math.random() < epsilon) {
      actions.push(Math.floor(Math.random() * config.actionDim))
    } else {
      actions.push(qValues[i][0] > qValues[i][1] ? 0 : 1)
    }
  }
  
  // Step all engines and collect transitions
  let transitionsAdded = 0
  let episodesCompleted = 0  // Count episodes completed in this batch
  
  for (let i = 0; i < n; i++) {
    const state = states[i]
    const action = actions[i]
    const result = engines[i].step(action as 0 | 1)
    
    // Add to replay buffer
    replayBuffer.add({
      state,
      action,
      reward: result.reward,
      nextState: result.observation,
      done: result.done,
    })
    transitionsAdded++
    experienceCount++
    fastTotalSteps++
    fastStepsSinceLastMetric++
    
    // Track episode stats
    fastEpisodeRewards[i] += result.reward
    fastEpisodeLengths[i]++
    
    // Handle episode end
    if (result.done) {
      episodesCompleted++  // Count completed episodes
      fastLastCompletedReward = fastEpisodeRewards[i]
      fastLastCompletedLength = fastEpisodeLengths[i]
      
      fastRecentRewards.push(fastEpisodeRewards[i])
      fastRecentLengths.push(fastEpisodeLengths[i])
      if (fastRecentRewards.length > METRICS_WINDOW) {
        fastRecentRewards.shift()
        fastRecentLengths.shift()
      }
      
      // Reset this bird
      engines[i].reset()
      fastEpisodeRewards[i] = 0
      fastEpisodeLengths[i] = 0

    }
  }
  
  // Update episode count once per batch (not per bird)
  // This prevents the episode counter from jumping by hundreds
  fastEpisode += episodesCompleted
  
  if (
    autoEvalEnabled &&
    !autoEvalRunning &&
    !isInWarmup() &&
    episodesCompleted > 0 &&
    fastEpisode - lastAutoEvalEpisode >= autoEvalInterval
  ) {
    runAutoEval()
    return transitionsAdded
  }
  
  return transitionsAdded
}

/**
 * Train on a batch from replay buffer
 */
let trainCallCount = 0
let lastTargetUpdateEnvStep = 0  // Track when we last updated target network

function trainOnBatch(): void {
  if (!policyNetwork || !targetNetwork || !replayBuffer || !config) return
  if (!replayBuffer.canSample(config.batchSize)) return
  
  const batch = replayBuffer.sample(config.batchSize)
  const states: number[][] = []
  const actions: number[] = []
  const targets: number[] = []
  
  // Compute targets using target network (batch)
  const nextStates = batch.map(t => t.nextState)
  const nextQValues = targetNetwork.predictBatch(nextStates)
  
  for (let i = 0; i < batch.length; i++) {
    const t = batch[i]
    states.push(t.state)
    actions.push(t.action)
    
    const maxNextQ = Math.max(...nextQValues[i])
    const target = t.reward + (t.done ? 0 : config.gamma * maxNextQ)
    targets.push(target)
  }
  
  // Train (batch GPU operation)
  const loss = policyNetwork.trainBatch(states, actions, targets)
  lastLoss = loss
  steps++
  trainCallCount++
  
  // Log training progress periodically
  if (trainCallCount % 500 === 0) {
    console.log(`[GPU Worker] Training step ${trainCallCount}, loss: ${loss.toFixed(6)}, buffer: ${replayBuffer.size()}, eps: ${epsilon.toFixed(3)}`)
  }
  
  // Periodically sync weights back to main thread so the UI agent sees progress
  if (trainCallCount % WEIGHT_SYNC_INTERVAL === 0) {
    const weights = policyNetwork.toJSON()
    self.postMessage({
      type: 'weights',
      data: weights,
      steps,
      loss: lastLoss,
    } as GPUWorkerResponse)
  }

  // Update target network based on ENV STEPS (not training steps)
  // This ensures consistent target update frequency regardless of numBirds
  if (fastTotalSteps - lastTargetUpdateEnvStep >= TARGET_UPDATE_ENV_STEPS) {
    targetNetwork.copyWeightsFrom(policyNetwork)
    lastTargetUpdateEnvStep = fastTotalSteps
    console.log(`[GPU Worker] Target network updated at env step ${fastTotalSteps}`)
  }
}

// History arrays for charting (accumulated over time)
let rewardHistoryForChart: number[] = []
let avgRewardHistoryForChart: number[] = []
const MAX_CHART_HISTORY = 200  // Keep last 200 data points for charts

function emitFastMetrics(): void {
  if (!config || !replayBuffer) return
  
  const avgReward = fastRecentRewards.length > 0
    ? fastRecentRewards.reduce((a, b) => a + b, 0) / fastRecentRewards.length
    : 0
  
  const avgLength = fastRecentLengths.length > 0
    ? fastRecentLengths.reduce((a, b) => a + b, 0) / fastRecentLengths.length
    : 0
  
  // Update chart histories - add current values
  if (fastEpisode > 0) {
    // Only add if we have new data (episode changed)
    const lastRewardInHistory = rewardHistoryForChart[rewardHistoryForChart.length - 1]
    if (rewardHistoryForChart.length === 0 || fastLastCompletedReward !== lastRewardInHistory) {
      rewardHistoryForChart.push(fastLastCompletedReward)
      avgRewardHistoryForChart.push(avgReward)
      
      // Trim to max length
      if (rewardHistoryForChart.length > MAX_CHART_HISTORY) {
        rewardHistoryForChart.shift()
        avgRewardHistoryForChart.shift()
      }
    }
  }
  
  const metrics: TrainingMetrics = {
    episode: fastEpisode,
    episodeReward: fastLastCompletedReward,
    episodeLength: fastLastCompletedLength,
    avgReward,
    avgLength,
    epsilon,
    loss: lastLoss,
    bufferSize: replayBuffer.size(),
    stepsPerSecond: fastStepsPerSecond,
    totalSteps: fastTotalSteps,
    isWarmup: isInWarmup(),
    learningRate: policyNetwork?.getLearningRate() ?? 0.0005,
    numBirds,
    batchSize: config.batchSize,
    trainSteps: trainCallCount,
    // Include chart histories for UI
    recentRewards: [...rewardHistoryForChart],
    recentAvgRewards: [...avgRewardHistoryForChart],
  }

  if (autoEvalRunning) {
    metrics.isAutoEval = true
    metrics.autoEvalTrial = autoEvalCurrentTrial
    metrics.autoEvalTrials = autoEvalTrials
  }
  
  self.postMessage({ type: 'fastMetrics', metrics } as GPUWorkerResponse)
}

/**
 * Main fast training loop
 * Optimized for multi-bird GPU training with adaptive training frequency
 */
function runFastModeBatch(): void {
  if (!fastModeRunning || !config || !policyNetwork || !replayBuffer) return
  
  const startTime = performance.now()
  
  // For very large numBirds, process in chunks to avoid blocking
  // But allow more steps when we have many birds (they're fast to simulate)
  const stepsPerBatch = numBirds >= 1000 
    ? Math.min(FAST_BATCH_STEPS * 2, Math.ceil(20000 / numBirds))  // More steps for high bird count
    : Math.min(FAST_BATCH_STEPS, Math.ceil(10000 / numBirds))
  
  // Adaptive time limit: allow more time when we have many birds (they're fast)
  const maxTimeMs = numBirds >= 1000 ? 100 : 50  // 100ms for 1000+ birds, 50ms otherwise
  
  let trainCounter = 0
  let expCollected = 0
  
  for (let step = 0; step < stepsPerBatch && fastModeRunning; step++) {
    // Step all birds (batch inference + env steps)
    const added = stepAllBirds()
    expCollected += added
    pendingExperienceDebt += added
    
    // Train periodically based on adaptive frequency
    const warmup = isInWarmup()
    const canSample = replayBuffer && replayBuffer.canSample(config.batchSize)
    
    if (!warmup && canSample && pendingExperienceDebt >= TARGET_TRAIN_RATIO) {
      const trainBudgetStart = performance.now()
      let iterations = 0
      
      while (
        pendingExperienceDebt >= TARGET_TRAIN_RATIO &&
        replayBuffer.canSample(config.batchSize) &&
        iterations < MAX_TRAIN_BATCHES_PER_LOOP
      ) {
        trainOnBatch()
        trainCounter++
        iterations++
        pendingExperienceDebt -= TARGET_TRAIN_RATIO
        
        if (performance.now() - trainBudgetStart > TRAIN_TIME_BUDGET_MS) {
          break
        }
      }
      
      if (iterations > 0) {
        updateEpsilon()
      }
    }
    
    // Don't block for too long (adaptive time limit)
    if (performance.now() - startTime > maxTimeMs) break
  }
  
  // Log training progress occasionally
  if (trainCounter > 0 && trainCallCount % 1000 === 0) {
    const ratio = expCollected / (trainCounter || 1)
    console.log(`[GPU Worker] Training: ${trainCallCount} batches, loss: ${lastLoss.toFixed(6)}, exp/train ratio: ${ratio.toFixed(1)}`)
  }
  
  // Prima di emettere le metriche, se è in corso un'auto-eval portiamola avanti
  // di un piccolo passo non bloccante (batch inferenza + pochi step).
  if (autoEvalRunning) {
    runAutoEvalBatch()
  }

  // Emit metrics
  const now = performance.now()
  const metricsInterval = isInWarmup() ? 100 : 500
  if (now - fastLastMetricsTime >= metricsInterval) {
    fastStepsPerSecond = (fastStepsSinceLastMetric / (now - fastLastMetricsTime)) * 1000
    fastStepsSinceLastMetric = 0
    fastLastMetricsTime = now
    emitFastMetrics()
  }
  
  // Continue loop (anche se c'è un'auto-eval in corso: i due loop si alternano)
  if (fastModeRunning) {
    setTimeout(runFastModeBatch, 0)
  }
}

function startFastMode(startingEpisode: number = 0, startingTotalSteps: number = 0): void {
  if (!config) return
  
  // Initialize engines if not already done
  if (engines.length !== numBirds) {
    initializeEngines(numBirds)
  } else {
    // Reset all engines
    for (let i = 0; i < engines.length; i++) {
      engines[i].reset()
      fastEpisodeRewards[i] = 0
      fastEpisodeLengths[i] = 0
    }
  }
  
  fastModeRunning = true
  fastEpisode = startingEpisode
  fastLastCompletedReward = 0
  fastLastCompletedLength = 0
  fastRecentRewards = []
  fastRecentLengths = []
  fastLastMetricsTime = performance.now()
  fastStepsSinceLastMetric = 0
  fastStepsPerSecond = 0
  fastTotalSteps = startingTotalSteps
  pendingExperienceDebt = 0
  lastAutoEvalEpisode = startingEpisode
  
  // Reset chart histories when starting fresh
  if (startingEpisode === 0) {
    rewardHistoryForChart = []
    avgRewardHistoryForChart = []
  }
  
  console.log(`[GPU Worker] Starting fast mode with ${numBirds} birds`)
  emitFastMetrics()
  runFastModeBatch()
}

function stopFastMode(): void {
  fastModeRunning = false
  pendingExperienceDebt = 0
  console.log('[GPU Worker] Stopped fast mode')
}

// ============================================================================
// Message Handler
// ============================================================================

self.onmessage = async (e: MessageEvent<GPUWorkerMessage>) => {
  try {
    const message = e.data
    
    switch (message.type) {
      case 'init': {
        config = message.config
        numBirds = config.numBirds || 1
        steps = 0
        epsilon = config.epsilonStart
        decayStartEpsilon = config.epsilonStart
        decayStartTrainStep = 0
        lastTargetUpdateEnvStep = 0
        
        // Auto-set optimal batch size based on numBirds
        config.batchSize = getOptimalBatchSize(numBirds)
        console.log(`[GPU Worker] Using batchSize ${config.batchSize} for ${numBirds} birds`)
        
        // Initialize TensorFlow.js with best available backend
        const { backend, gpuAvailable } = await initTensorFlow()
        
        // Create networks
        policyNetwork = createDQNNetworkTF(
          config.inputDim,
          config.hiddenLayers,
          config.actionDim,
          config.learningRate
        )
        
        targetNetwork = createDQNNetworkTF(
          config.inputDim,
          config.hiddenLayers,
          config.actionDim,
          config.learningRate
        )
        
        targetNetwork.copyWeightsFrom(policyNetwork)
        
        // Create replay buffer
        replayBuffer = new ReplayBuffer(config.bufferSize)
        
        // Initialize bird engines
        initializeEngines(numBirds)
        
        // Send ready with backend info
        self.postMessage({ type: 'ready', backend, gpuAvailable } as GPUWorkerResponse)
        
        // Send initial weights
        const initialWeights = policyNetwork.toJSON()
        self.postMessage({
          type: 'weights',
          data: initialWeights,
          steps: 0,
          loss: 0,
        } as GPUWorkerResponse)
        
        console.log(`[GPU Worker] Initialized with ${backend} backend, ${numBirds} birds`)
        break
      }
      
      case 'setNumBirds': {
        const newNumBirds = Math.max(1, Math.min(10000, message.value))
        if (newNumBirds !== numBirds) {
          numBirds = newNumBirds
          if (fastModeRunning) {
            // Reinitialize engines with new count
            initializeEngines(numBirds)
          }
          
          // Auto-adjust batch size for optimal GPU utilization
          // More birds = larger batches for better GPU efficiency
          if (config) {
            const optimalBatch = getOptimalBatchSize(numBirds)
            if (config.batchSize !== optimalBatch) {
              config.batchSize = optimalBatch
              console.log(`[GPU Worker] Auto-adjusted batchSize to ${optimalBatch} for ${numBirds} birds`)
            }
          }
          
          console.log(`[GPU Worker] Set numBirds to ${numBirds}`)
        }
        break
      }
      
      case 'setBatchSize': {
        if (config) {
          const newBatchSize = Math.max(32, Math.min(4096, message.value))
          config.batchSize = newBatchSize
          console.log(`[GPU Worker] Set batchSize to ${newBatchSize}`)
        }
        break
      }
      
      case 'experience': {
        if (!replayBuffer) break
        replayBuffer.add(message.transition)
        experienceCount++
        
        if (!isInWarmup() && experienceCount % 8 === 0 && replayBuffer.canSample(config?.batchSize || 64)) {
          trainOnBatch()
          updateEpsilon()
        }
        break
      }
      
      case 'requestWeights': {
        if (!policyNetwork) break
        const weights = policyNetwork.toJSON()
        self.postMessage({
          type: 'weights',
          data: weights,
          steps,
          loss: lastLoss,
        } as GPUWorkerResponse)
        break
      }
      
      case 'setWeights': {
        if (!policyNetwork || !targetNetwork) break
        policyNetwork.loadJSON(message.data)
        targetNetwork.loadJSON(message.data)
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
          decayStartTrainStep = trainCallCount
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
        // Update all engines
        for (const engine of engines) {
          engine.setRewardConfig(message.config)
        }
        break
      }

      case 'setAutoEval': {
        autoEvalEnabled = message.enabled
        if (typeof message.interval === 'number') {
          autoEvalInterval = Math.max(1, message.interval)
        }
        if (typeof message.trials === 'number') {
          autoEvalTrials = Math.max(1, message.trials)
        }
        console.log(`[GPU Worker] Auto-eval: ${autoEvalEnabled ? 'enabled' : 'disabled'} (every ${autoEvalInterval} eps, ${autoEvalTrials} trials)`)
        if (!autoEvalEnabled && autoEvalRunning) {
          finishAutoEval()
        }
        break
      }
      
      case 'startFast': {
        if (!policyNetwork || !replayBuffer) {
          console.warn('[GPU Worker] Cannot start fast mode without initialization')
          break
        }
        startFastMode(message.startingEpisode, message.startingTotalSteps)
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
        decayStartTrainStep = 0
        lastTargetUpdateEnvStep = 0
        lastLoss = 0
        trainCallCount = 0

        // Reset auto-eval state
        autoEvalRunning = false
        autoEvalCurrentTrial = 0
        autoEvalScores = []
        autoEvalEngines = []
        autoEvalEpisodeScores = []
        lastAutoEvalEpisode = 0
        
        // Reset chart histories
        rewardHistoryForChart = []
        avgRewardHistoryForChart = []
        
        replayBuffer?.clear()
        
        if (config) {
          policyNetwork?.dispose()
          targetNetwork?.dispose()
          
          policyNetwork = createDQNNetworkTF(
            config.inputDim,
            config.hiddenLayers,
            config.actionDim,
            config.learningRate
          )
          
          targetNetwork = createDQNNetworkTF(
            config.inputDim,
            config.hiddenLayers,
            config.actionDim,
            config.learningRate
          )
          
          targetNetwork.copyWeightsFrom(policyNetwork)
          
          const resetWeights = policyNetwork.toJSON()
          self.postMessage({
            type: 'weights',
            data: resetWeights,
            steps: 0,
            loss: 0,
          } as GPUWorkerResponse)
        }
        
        // Reset engines
        initializeEngines(numBirds)
        break
      }
      
      case 'dispose': {
        stopFastMode()
        policyNetwork?.dispose()
        targetNetwork?.dispose()
        policyNetwork = null
        targetNetwork = null
        replayBuffer = null
        engines = []
        break
      }
      
      default: {
        console.warn('[GPU Worker] Unknown message type:', (message as any).type)
      }
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
    } as GPUWorkerResponse)
  }
}

export type { GPUWorkerMessage, GPUWorkerResponse }

