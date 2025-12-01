/**
 * Simple Neural Network implementation in pure TypeScript
 * No external dependencies - just matrix operations
 */

export interface LayerConfig {
  inputSize: number
  outputSize: number
  activation: 'relu' | 'linear'
}

export interface NetworkConfig {
  layers: LayerConfig[]
  learningRate: number
}

interface Layer {
  weights: number[][]  // [inputSize][outputSize]
  biases: number[]     // [outputSize]
  activation: 'relu' | 'linear'
  // For backprop
  lastInput?: number[]
  lastOutput?: number[]
  lastPreActivation?: number[]
}

export class NeuralNetwork {
  private layers: Layer[] = []
  private learningRate: number

  constructor(config: NetworkConfig) {
    this.learningRate = config.learningRate
    
    for (const layerConfig of config.layers) {
      this.layers.push({
        weights: this.initWeights(layerConfig.inputSize, layerConfig.outputSize),
        biases: new Array(layerConfig.outputSize).fill(0).map(() => (Math.random() - 0.5) * 0.1),
        activation: layerConfig.activation,
      })
    }
  }

  /**
   * Xavier/Glorot initialization
   */
  private initWeights(inputSize: number, outputSize: number): number[][] {
    const scale = Math.sqrt(2.0 / (inputSize + outputSize))
    const weights: number[][] = []
    
    for (let i = 0; i < inputSize; i++) {
      weights[i] = []
      for (let j = 0; j < outputSize; j++) {
        weights[i][j] = (Math.random() * 2 - 1) * scale
      }
    }
    
    return weights
  }

  /**
   * Forward pass - returns output
   */
  forward(input: number[]): number[] {
    let current = input

    // Check for NaN in input
    if (input.some(x => !Number.isFinite(x))) {
      console.warn('[NN] NaN/Inf detected in input, returning zeros')
      return new Array(this.layers[this.layers.length - 1].biases.length).fill(0)
    }

    for (const layer of this.layers) {
      layer.lastInput = [...current]
      
      // Matrix multiplication: output = input @ weights + biases
      const preActivation: number[] = []
      for (let j = 0; j < layer.biases.length; j++) {
        let sum = layer.biases[j]
        for (let i = 0; i < current.length; i++) {
          sum += current[i] * layer.weights[i][j]
        }
        // Clamp to prevent overflow
        preActivation[j] = Math.max(-1e6, Math.min(1e6, sum))
      }
      
      layer.lastPreActivation = preActivation
      
      // Apply activation
      const output = preActivation.map(x => 
        layer.activation === 'relu' ? Math.max(0, x) : x
      )
      
      layer.lastOutput = output
      current = output
    }

    // Final NaN check
    if (current.some(x => !Number.isFinite(x))) {
      console.warn('[NN] NaN/Inf detected in output, resetting weights')
      this.resetWeights()
      return new Array(current.length).fill(0)
    }

    return current
  }

  /**
   * Reset weights if NaN detected
   */
  private resetWeights(): void {
    for (const layer of this.layers) {
      const inputSize = layer.weights.length
      const outputSize = layer.biases.length
      layer.weights = this.initWeights(inputSize, outputSize)
      layer.biases = new Array(outputSize).fill(0).map(() => (Math.random() - 0.5) * 0.1)
    }
  }

  /**
   * Predict Q-values for a state
   */
  predict(state: number[]): number[] {
    return this.forward(state)
  }

  private static readonly GRAD_CLIP = 1.0  // Gradient clipping threshold
  private static readonly WEIGHT_CLIP = 10.0 // Max weight magnitude

  /**
   * Train on a single sample using gradient descent
   * Uses Huber loss (smooth L1) with gradient clipping
   */
  trainStep(
    state: number[],
    action: number,
    target: number
  ): number {
    // Check for invalid target
    if (!Number.isFinite(target)) {
      console.warn('[NN] Invalid target value:', target)
      return 0
    }

    // Forward pass
    const qValues = this.forward(state)
    
    // Compute loss (only for the action taken)
    const predicted = qValues[action]
    const error = target - predicted
    
    // Skip if error is invalid
    if (!Number.isFinite(error)) {
      console.warn('[NN] Invalid error in training')
      return 0
    }
    
    const loss = Math.abs(error) < 1 
      ? 0.5 * error * error 
      : Math.abs(error) - 0.5

    // Backward pass
    // Gradient of Huber loss (already clipped to [-1, 1] by Huber)
    let gradOutput = Math.abs(error) < 1 ? -error : -Math.sign(error)
    
    // Only backprop through the action we took
    const outputGrad = new Array(qValues.length).fill(0)
    outputGrad[action] = gradOutput

    // Backprop through layers (reverse order)
    let currentGrad = outputGrad
    
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer = this.layers[l]
      const input = layer.lastInput!
      const preAct = layer.lastPreActivation!
      
      // Gradient through activation
      const gradPreAct = currentGrad.map((g, i) => {
        if (layer.activation === 'relu') {
          return preAct[i] > 0 ? g : 0
        }
        return g // linear
      })
      
      // Clip gradients
      const clippedGradPreAct = gradPreAct.map(g => 
        Math.max(-NeuralNetwork.GRAD_CLIP, Math.min(NeuralNetwork.GRAD_CLIP, g))
      )
      
      // Gradient for biases
      for (let j = 0; j < layer.biases.length; j++) {
        layer.biases[j] -= this.learningRate * clippedGradPreAct[j]
        // Clip biases
        layer.biases[j] = Math.max(-NeuralNetwork.WEIGHT_CLIP, Math.min(NeuralNetwork.WEIGHT_CLIP, layer.biases[j]))
      }
      
      // Gradient for weights and compute input gradient
      const inputGrad = new Array(input.length).fill(0)
      
      for (let i = 0; i < input.length; i++) {
        for (let j = 0; j < layer.biases.length; j++) {
          // Weight gradient (with input clipping too)
          const weightGrad = input[i] * clippedGradPreAct[j]

          // Cache current weight for backprop before applying the update
          const currentWeight = layer.weights[i][j]

          // Input gradient for next layer should use the weight *before* the update
          inputGrad[i] += currentWeight * clippedGradPreAct[j]

          // Apply weight update
          layer.weights[i][j] = currentWeight - this.learningRate * weightGrad
          // Clip weights
          layer.weights[i][j] = Math.max(-NeuralNetwork.WEIGHT_CLIP, Math.min(NeuralNetwork.WEIGHT_CLIP, layer.weights[i][j]))
        }
      }
      
      // Clip input gradient for next layer
      currentGrad = inputGrad.map(g => 
        Math.max(-NeuralNetwork.GRAD_CLIP, Math.min(NeuralNetwork.GRAD_CLIP, g))
      )
    }

    return loss
  }

  /**
   * Train on a batch of samples
   */
  trainBatch(
    states: number[][],
    actions: number[],
    targets: number[]
  ): number {
    let totalLoss = 0
    
    for (let i = 0; i < states.length; i++) {
      totalLoss += this.trainStep(states[i], actions[i], targets[i])
    }
    
    return totalLoss / states.length
  }

  /**
   * Get all weights for visualization
   */
  getWeights(): number[][][] {
    return this.layers.map(l => l.weights)
  }

  /**
   * Get all biases
   */
  getBiases(): number[][] {
    return this.layers.map(l => l.biases)
  }

  /**
   * Get activations from last forward pass
   */
  getActivations(): number[][] {
    const activations: number[][] = []
    
    if (this.layers[0].lastInput) {
      activations.push(this.layers[0].lastInput)
    }
    
    for (const layer of this.layers) {
      if (layer.lastOutput) {
        activations.push(layer.lastOutput)
      }
    }
    
    return activations
  }

  /**
   * Get layer info for visualization
   */
  getLayerSizes(): number[] {
    const sizes: number[] = []
    
    if (this.layers.length > 0) {
      sizes.push(this.layers[0].weights.length) // input size
    }
    
    for (const layer of this.layers) {
      sizes.push(layer.biases.length)
    }
    
    return sizes
  }

  /**
   * Copy weights from another network
   */
  copyWeightsFrom(source: NeuralNetwork): void {
    const sourceWeights = source.getWeights()
    const sourceBiases = source.getBiases()
    
    for (let l = 0; l < this.layers.length; l++) {
      // Deep copy weights
      for (let i = 0; i < this.layers[l].weights.length; i++) {
        for (let j = 0; j < this.layers[l].weights[i].length; j++) {
          this.layers[l].weights[i][j] = sourceWeights[l][i][j]
        }
      }
      
      // Deep copy biases
      for (let j = 0; j < this.layers[l].biases.length; j++) {
        this.layers[l].biases[j] = sourceBiases[l][j]
      }
    }
  }

  /**
   * Set learning rate
   */
  setLearningRate(lr: number): void {
    this.learningRate = lr
  }

  /**
   * Serialize weights to JSON
   */
  toJSON(): { weights: number[][][], biases: number[][] } {
    return {
      // Return deep copies so callers can't mutate internal state
      weights: this.layers.map(layer => layer.weights.map(row => [...row])),
      biases: this.layers.map(layer => [...layer.biases]),
    }
  }

  /**
   * Load weights from JSON
   */
  loadJSON(data: { weights: number[][][], biases: number[][] }): void {
    for (let l = 0; l < this.layers.length; l++) {
      // Deep copy to avoid shared references between networks
      this.layers[l].weights = data.weights[l].map(row => [...row])
      this.layers[l].biases = [...data.biases[l]]
    }
  }
}

/**
 * Create a DQN-style network
 */
export function createDQNNetwork(
  inputDim: number,
  hiddenLayers: number[],
  outputDim: number,
  learningRate: number
): NeuralNetwork {
  const layers: LayerConfig[] = []
  
  // First hidden layer
  layers.push({
    inputSize: inputDim,
    outputSize: hiddenLayers[0],
    activation: 'relu',
  })
  
  // Additional hidden layers
  for (let i = 1; i < hiddenLayers.length; i++) {
    layers.push({
      inputSize: hiddenLayers[i - 1],
      outputSize: hiddenLayers[i],
      activation: 'relu',
    })
  }
  
  // Output layer
  layers.push({
    inputSize: hiddenLayers[hiddenLayers.length - 1],
    outputSize: outputDim,
    activation: 'linear',
  })
  
  return new NeuralNetwork({ layers, learningRate })
}

