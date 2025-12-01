import { describe, expect, it } from 'vitest'
import { NeuralNetwork } from './NeuralNetwork'

function setLayer(network: NeuralNetwork, index: number, weights: number[][], biases: number[]) {
  const layers = (network as unknown as { layers: any[] }).layers
  layers[index].weights = weights.map(row => [...row])
  layers[index].biases = [...biases]
}

describe('NeuralNetwork', () => {
  it('performs a forward pass with the expected activations', () => {
    const network = new NeuralNetwork({
      learningRate: 0.1,
      layers: [
        { inputSize: 2, outputSize: 2, activation: 'relu' },
        { inputSize: 2, outputSize: 1, activation: 'linear' },
      ],
    })

    setLayer(network, 0, [[0.1, -0.2], [0, 0.3]], [0, 0])
    setLayer(network, 1, [[0.4], [-0.1]], [0])

    const output = network.predict([1, 2])

    expect(output).toHaveLength(1)
    expect(output[0]).toBeCloseTo(0, 6)
  })

  it('applies Huber loss gradients and uses pre-update weights for backprop', () => {
    const network = new NeuralNetwork({
      learningRate: 0.1,
      layers: [
        { inputSize: 2, outputSize: 2, activation: 'relu' },
        { inputSize: 2, outputSize: 1, activation: 'linear' },
      ],
    })

    setLayer(network, 0, [[0.1, -0.2], [0, 0.3]], [0, 0])
    setLayer(network, 1, [[0.4], [-0.1]], [0])

    const loss = network.trainStep([1, 2], 0, 1)

    expect(loss).toBeCloseTo(0.5, 6)

    const layers = network.getWeights()
    const biases = network.getBiases()

    // Layer 1 (input -> hidden) updates
    expect(layers[0][0][0]).toBeCloseTo(0.14, 6)
    expect(layers[0][1][0]).toBeCloseTo(0.08, 6)
    expect(layers[0][0][1]).toBeCloseTo(-0.21, 6)
    expect(layers[0][1][1]).toBeCloseTo(0.28, 6)

    expect(biases[0][0]).toBeCloseTo(0.04, 6)
    expect(biases[0][1]).toBeCloseTo(-0.01, 6)

    // Layer 2 (hidden -> output) updates
    expect(layers[1][0][0]).toBeCloseTo(0.41, 6)
    expect(layers[1][1][0]).toBeCloseTo(-0.06, 6)
    expect(biases[1][0]).toBeCloseTo(0.1, 6)
  })

  it('deep-copies weights and biases when serializing and loading', () => {
    const config = {
      learningRate: 0.1,
      layers: [
        { inputSize: 2, outputSize: 1, activation: 'linear' },
      ],
    }

    const source = new NeuralNetwork(config)
    setLayer(source, 0, [[0.25], [-0.5]], [0.75])

    const snapshot = source.toJSON()
    const loaded = new NeuralNetwork(config)
    loaded.loadJSON(snapshot)

    // Mutate the snapshot and ensure the loaded network remains unchanged
    snapshot.weights[0][0][0] = 999
    snapshot.biases[0][0] = 999

    const loadedWeights = loaded.getWeights()
    const loadedBiases = loaded.getBiases()

    expect(loadedWeights[0][0][0]).toBeCloseTo(0.25, 6)
    expect(loadedWeights[0][1][0]).toBeCloseTo(-0.5, 6)
    expect(loadedBiases[0][0]).toBeCloseTo(0.75, 6)

    // Changing the loaded network should not leak back to the source
    loadedWeights[0][0][0] = -123
    loadedBiases[0][0] = -123

    const sourceWeights = source.getWeights()
    const sourceBiases = source.getBiases()

    expect(sourceWeights[0][0][0]).toBeCloseTo(0.25, 6)
    expect(sourceBiases[0][0]).toBeCloseTo(0.75, 6)
  })

  it('clips large gradients and weight updates to avoid explosions', () => {
    const network = new NeuralNetwork({
      learningRate: 1,
      layers: [
        { inputSize: 2, outputSize: 1, activation: 'linear' },
      ],
    })

    setLayer(network, 0, [[9.9], [0.5]], [0])

    // Large positive target so the update would overshoot without clipping
    network.trainStep([1, 1], 0, 20)

    const [weights] = network.getWeights()
    const [biases] = network.getBiases()

    // Gradient should be clipped to -1, leading to +1 weight update that gets clipped to 10
    expect(weights[0][0]).toBeCloseTo(10, 6)
    // Second weight and bias also respect the gradient and weight clip values
    expect(weights[1][0]).toBeCloseTo(1.5, 6)
    expect(biases[0]).toBeCloseTo(1, 6)
  })

  it('stops gradients at ReLU boundaries so preceding weights remain unchanged', () => {
    const network = new NeuralNetwork({
      learningRate: 0.1,
      layers: [
        { inputSize: 1, outputSize: 1, activation: 'relu' },
        { inputSize: 1, outputSize: 1, activation: 'linear' },
      ],
    })

    setLayer(network, 0, [[-1]], [0]) // Ensures negative pre-activation for positive inputs
    setLayer(network, 1, [[1]], [0])

    const preTrainWeights = network.getWeights().map(layer => layer.map(row => [...row]))

    // Positive state yields negative pre-activation in first layer -> ReLU outputs zero -> no gradient flows back
    network.trainStep([1], 0, 1)

    const postWeights = network.getWeights()
    const postBiases = network.getBiases()

    // Hidden-layer weight remains unchanged because the ReLU gradient is zero
    expect(postWeights[0][0][0]).toBeCloseTo(preTrainWeights[0][0][0], 6)
    // Output layer receives zero input, so its weight stays the same but bias adjusts to reduce the loss
    expect(postWeights[1][0][0]).toBeCloseTo(preTrainWeights[1][0][0], 6)
    expect(postBiases[1][0]).toBeGreaterThan(0)
  })

  it('returns zeros and preserves weights when NaN or Infinite inputs are provided', () => {
    const network = new NeuralNetwork({
      learningRate: 0.1,
      layers: [
        { inputSize: 2, outputSize: 1, activation: 'linear' },
      ],
    })

    const initialWeights = network.getWeights().map(layer => layer.map(row => [...row]))

    const nanOutput = network.predict([NaN, 1])
    const infOutput = network.predict([Infinity, -Infinity])

    expect(nanOutput).toEqual([0])
    expect(infOutput).toEqual([0])

    // Ensure weights are unchanged by the invalid inputs
    const finalWeights = network.getWeights()
    expect(finalWeights[0][0][0]).toBeCloseTo(initialWeights[0][0][0], 6)
    expect(finalWeights[0][1][0]).toBeCloseTo(initialWeights[0][1][0], 6)
  })
})
