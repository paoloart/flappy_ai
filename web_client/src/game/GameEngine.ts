/**
 * Core game engine for Flappy Bird
 * Handles physics, collision detection, and game logic
 */

import {
  GameConfig,
  type GameAction,
  type RewardConfig,
  DefaultRewardConfig,
  type ObservationConfig,
  DefaultObservationConfig,
} from './config'
import {
  type RawGameState,
  type PipeState,
  createInitialState,
  stateToObservation,
  clamp,
} from './GameState'

export interface StepResult {
  observation: number[]
  reward: number
  done: boolean
  info: {
    score: number
    episode: number
    steps: number
  }
}

export class GameEngine {
  private state: RawGameState
  private rewardConfig: RewardConfig
  private observationConfig: ObservationConfig
  private prevScore: number = 0
  private episode: number = 0
  private totalSteps: number = 0

  constructor(
    rewardConfig: RewardConfig = DefaultRewardConfig,
    observationConfig: ObservationConfig = DefaultObservationConfig
  ) {
    this.rewardConfig = rewardConfig
    this.observationConfig = observationConfig
    this.state = createInitialState()
  }

  /**
   * Reset the game and return initial observation
   */
  reset(): number[] {
    this.state = createInitialState()
    this.prevScore = 0
    this.episode++

    // Give the bird an initial upward velocity (like Python version)
    this.state.birdVelY = GameConfig.BIRD.FLAP_VELOCITY
    this.state.birdRotation = 80

    // Spawn initial pipes
    this.spawnInitialPipes()

    return this.getObservation()
  }

  /**
   * Execute one game step with the given action
   */
  step(action: GameAction): StepResult {
    if (this.state.done) {
      return {
        observation: this.getObservation(),
        reward: 0,
        done: true,
        info: this.getInfo(),
      }
    }

    // Apply action (flap)
    if (action === 1) {
      this.flap()
    }

    // Update physics
    this.updateBird()
    this.updatePipes()

    // Check collisions
    this.checkCollisions()

    // Update score
    this.updateScore()

    // Calculate reward
    const reward = this.calculateReward(action)
    this.prevScore = this.state.score

    this.state.frameCount++
    this.totalSteps++

    return {
      observation: this.getObservation(),
      reward,
      done: this.state.done,
      info: this.getInfo(),
    }
  }

  /**
   * Get the current observation vector
   */
  getObservation(): number[] {
    return stateToObservation(this.state, this.observationConfig)
  }

  /**
   * Get raw game state (for rendering)
   */
  getState(): Readonly<RawGameState> {
    return this.state
  }

  /**
   * Get episode info
   */
  getInfo() {
    return {
      score: this.state.score,
      episode: this.episode,
      steps: this.totalSteps,
    }
  }

  /**
   * Update reward configuration (live during training)
   */
  setRewardConfig(config: Partial<RewardConfig>): void {
    this.rewardConfig = { ...this.rewardConfig, ...config }
  }

  // ===== Private methods =====

  private flap(): void {
    if (this.state.birdY > GameConfig.BIRD.MIN_Y) {
      this.state.birdVelY = GameConfig.BIRD.FLAP_VELOCITY
      this.state.birdRotation = 80 // Tilt up on flap
    }
  }

  private updateBird(): void {
    // Apply gravity
    if (this.state.birdVelY < GameConfig.BIRD.MAX_VELOCITY_DOWN) {
      this.state.birdVelY += GameConfig.BIRD.GRAVITY
    }

    // Update position
    this.state.birdY = clamp(
      this.state.birdY + this.state.birdVelY,
      GameConfig.BIRD.MIN_Y,
      GameConfig.VIEWPORT_HEIGHT - GameConfig.BIRD.HEIGHT * 0.75
    )

    // Update rotation
    this.state.birdRotation = clamp(
      this.state.birdRotation + GameConfig.BIRD.ROTATION_SPEED,
      GameConfig.BIRD.ROTATION_MIN,
      GameConfig.BIRD.ROTATION_MAX
    )
  }

  private updatePipes(): void {
    // Move all pipes
    for (const pipe of this.state.pipes) {
      pipe.x += GameConfig.PIPE.VELOCITY
    }

    // Update floor scroll position (synced with pipe movement)
    // Floor sprite width is typically 336px, but we use 336 as a safe default
    const floorWidth = 336
    this.state.floorX = (this.state.floorX + GameConfig.FLOOR.VELOCITY) % floorWidth
    if (this.state.floorX > 0) this.state.floorX -= floorWidth // Keep it negative or zero

    // Remove off-screen pipes
    this.state.pipes = this.state.pipes.filter(
      (pipe) => pipe.x > -GameConfig.PIPE.WIDTH
    )

    // Spawn new pipes if needed
    this.maybeSpawnPipe()
  }

  private maybeSpawnPipe(): void {
    if (this.state.pipes.length === 0) {
      this.spawnPipe()
      return
    }

    const lastPipe = this.state.pipes[this.state.pipes.length - 1]
    const distanceFromLast = GameConfig.WIDTH - (lastPipe.x + GameConfig.PIPE.WIDTH)

    if (distanceFromLast > GameConfig.PIPE.WIDTH * 2.5) {
      this.spawnPipe()
    }
  }

  private spawnPipe(): void {
    // Random gap position (matching Python logic)
    const baseY = GameConfig.VIEWPORT_HEIGHT
    const minGapY = baseY * 0.2 + GameConfig.PIPE.GAP / 2
    const maxGapY = baseY * 0.8 - GameConfig.PIPE.GAP / 2
    const gapCenterY = minGapY + Math.random() * (maxGapY - minGapY)

    this.state.pipes.push({
      x: GameConfig.WIDTH + 10,
      gapCenterY,
      gapVelY: 0,
      passed: false,
    })
  }

  private spawnInitialPipes(): void {
    // Spawn first pipe
    const pipe1 = this.createRandomPipe()
    pipe1.x = GameConfig.PIPE.INITIAL_X_OFFSET

    // Spawn second pipe
    const pipe2 = this.createRandomPipe()
    pipe2.x = pipe1.x + GameConfig.PIPE.SPAWN_DISTANCE

    this.state.pipes = [pipe1, pipe2]
  }

  private createRandomPipe(): PipeState {
    const baseY = GameConfig.VIEWPORT_HEIGHT
    const minGapY = baseY * 0.2 + GameConfig.PIPE.GAP / 2
    const maxGapY = baseY * 0.8 - GameConfig.PIPE.GAP / 2
    const gapCenterY = minGapY + Math.random() * (maxGapY - minGapY)

    return {
      x: GameConfig.WIDTH + 10,
      gapCenterY,
      gapVelY: 0,
      passed: false,
    }
  }

  private checkCollisions(): void {
    const birdX = GameConfig.BIRD.X
    const birdY = this.state.birdY
    const birdW = GameConfig.BIRD.WIDTH
    const birdH = GameConfig.BIRD.HEIGHT

    // Floor collision
    if (birdY + birdH >= GameConfig.VIEWPORT_HEIGHT) {
      this.state.done = true
      return
    }

    // Ceiling collision (allow going above but not too far)
    if (birdY <= GameConfig.BIRD.MIN_Y) {
      // Not a death, just clamped
    }

    // Pipe collisions
    for (const pipe of this.state.pipes) {
      // Check if bird overlaps with pipe horizontally
      if (
        birdX + birdW > pipe.x &&
        birdX < pipe.x + GameConfig.PIPE.WIDTH
      ) {
        // Check vertical overlap with upper or lower pipe
        const gapTop = pipe.gapCenterY - GameConfig.PIPE.GAP / 2
        const gapBottom = pipe.gapCenterY + GameConfig.PIPE.GAP / 2

        if (birdY < gapTop || birdY + birdH > gapBottom) {
          this.state.done = true
          return
        }
      }
    }
  }

  private updateScore(): void {
    const birdCenterX = GameConfig.BIRD.X + GameConfig.BIRD.WIDTH / 2

    for (const pipe of this.state.pipes) {
      if (!pipe.passed) {
        const pipeCenterX = pipe.x + GameConfig.PIPE.WIDTH / 2

        // Check if bird just crossed pipe center
        if (birdCenterX >= pipeCenterX && birdCenterX < pipeCenterX - GameConfig.PIPE.VELOCITY) {
          pipe.passed = true
          this.state.score++
        }
      }
    }
  }

  private calculateReward(action: GameAction): number {
    let reward = 0

    // Score reward (passing pipes)
    const scoreDelta = this.state.score - this.prevScore
    reward += scoreDelta * this.rewardConfig.passPipe

    // Step penalty
    reward += this.rewardConfig.stepPenalty

    // Flap cost
    if (action === 1 && !this.state.done && this.rewardConfig.flapCost > 0) {
      reward -= this.rewardConfig.flapCost
    }

    // Death penalty
    if (this.state.done) {
      reward = this.rewardConfig.deathPenalty
    } else {
      // Out of bounds penalty (going above screen) - hardcoded like Python
      // This teaches the agent to stay within the visible viewport
      if (this.state.birdY < 0) {
        reward -= 0.1 // Fixed penalty for being above viewport
      }
    }

    // Dense reward shaping: reward for being close to gap center
    // At center: reward = centerReward, at max distance: reward = 0
    if (!this.state.done && this.rewardConfig.centerReward > 0) {
      const pipe1 = this.state.pipes[0]
      if (pipe1) {
        // Calculate distance from bird to gap center
        const distance = Math.abs(pipe1.gapCenterY - this.state.birdY)
        
        // Max distance is half the viewport (bird can be above or below center)
        const maxDistance = GameConfig.VIEWPORT_HEIGHT / 2
        
        // Proximity: 1.0 at center, 0.0 at max distance
        const proximity = Math.max(0, 1 - (distance / maxDistance))
        
        // Reward scales linearly with proximity
        // centerReward=0.5 at center → 0.5 per frame, at edge → 0 per frame
        reward += this.rewardConfig.centerReward * proximity
      }
    }

    return reward
  }
}

