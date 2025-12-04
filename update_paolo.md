# 🚀 GPU Training Update - Fork di Paolo

Questo documento elenca tutte le modifiche apportate alla fork originale di [giosilvi/flappy_ai](https://github.com/giosilvi/flappy_ai).

---

## 📋 Sommario delle Modifiche

### 1. **Supporto GPU con TensorFlow.js**

Aggiunto il supporto per l'addestramento accelerato via GPU utilizzando TensorFlow.js con backend WebGPU/WebGL.

**File modificati/aggiunti:**
- `web_client/package.json` - Aggiunte dipendenze:
  ```json
  "@tensorflow/tfjs": "^4.22.0",
  "@tensorflow/tfjs-backend-webgpu": "^4.22.0"
  ```
- `web_client/src/rl/NeuralNetworkTF.ts` (**NUOVO**) - Implementazione della rete neurale con TensorFlow.js
- `web_client/src/rl/GPUDQNAgent.ts` (**NUOVO**) - Agente DQN per GPU
- `web_client/src/rl/GPUTrainingLoop.ts` (**NUOVO**) - Training loop per GPU
- `web_client/src/rl/gpu.worker.ts` (**NUOVO**) - Web Worker per simulazione multi-bird e training GPU

**Caratteristiche:**
- Inizializzazione automatica del backend WebGPU (fallback su WebGL/CPU)
- Batch inference/training per efficienza GPU
- Gradient clipping per stabilità

---

### 2. **Simulazione Multi-Bird Parallela**

Aggiunta la possibilità di eseguire fino a 10.000 bird in parallelo per accelerare la raccolta di esperienze.

**File modificati:**
- `web_client/src/rl/gpu.worker.ts` - Gestione multi-bird con array di `GameEngine`
- `web_client/src/components/ControlPanel.vue` - Slider per numero di bird (1-10.000)
- `web_client/src/App.vue` - Gestione stato `numBirds`

**Caratteristiche:**
- Batch inference su tutti i bird contemporaneamente
- Ogni bird contribuisce al replay buffer
- Conteggio episodi aggregato per batch

---

### 3. **Auto-Evaluation Non Bloccante**

Refactoring dell'auto-valutazione per renderla non bloccante durante il training GPU.

**File modificati:**
- `web_client/src/rl/gpu.worker.ts`:
  - Auto-eval eseguita con più bird in parallelo (`AUTO_EVAL_PARALLEL_BIRDS = 16`)
  - Training e auto-eval si alternano senza bloccarsi
  - Default: ogni 5000 episodi, 20 trial
- `web_client/src/rl/training.worker.ts` - Allineato intervallo a 5000 episodi

**Parametri:**
| Parametro | Valore Default |
|-----------|---------------|
| `autoEvalInterval` | 5000 episodi |
| `autoEvalTrials` | 20 (GPU) / 100 (CPU) |

---

### 4. **Penalità Out of Bounds**

Aggiunta una penalità quando il bird va fuori dallo schermo (sopra o troppo vicino al pavimento).

**File modificati:**
- `web_client/src/game/config.ts`:
  ```typescript
  export interface RewardConfig {
    // ...
    outOfBoundsPenalty: number  // Default: -0.2
  }
  ```
- `web_client/src/game/GameEngine.ts` - Applicazione della penalità
- `web_client/src/components/ControlPanel.vue` - Slider "Out of Bounds" penalty

---

### 5. **UI Migliorata per GPU Training**

**File modificati:**
- `web_client/src/components/ControlPanel.vue`:
  - Toggle "GPU Training" con indicazione backend (webgpu/webgl)
  - Slider "Parallel Birds" con scala logaritmica
  - Presets rapidi (100, 500, 1000, 5000, 10000 birds)
  
- `web_client/src/components/MetricsPanel.vue`:
  - Sezione "GPU Training Active" con badge backend
  - Contatore bird e episodi
  - Indicatore auto-eval con barra di progresso
  - Grafici aggiornati anche in GPU fast mode

- `web_client/src/App.vue`:
  - Gestione stato GPU: `useGPU`, `gpuAvailable`, `gpuBackend`
  - Propagazione metriche dai worker ai componenti

---

### 6. **Sincronizzazione Metriche e Charts**

**File modificati:**
- `web_client/src/rl/gpu.worker.ts`:
  - Array `rewardHistoryForChart` e `avgRewardHistoryForChart` per i grafici
  - Emissione periodica con `recentRewards` e `recentAvgRewards`
  
- `web_client/src/rl/types.ts`:
  ```typescript
  export interface TrainingMetrics {
    // ... existing fields
    numBirds?: number
    gpuBackend?: string
    batchSize?: number
    trainSteps?: number
    recentRewards?: number[]
    recentAvgRewards?: number[]
    isAutoEval?: boolean
    autoEvalTrial?: number
    autoEvalTrials?: number
  }
  ```

---

### 7. **Configurazione ngrok per Testing**

**File modificato:**
- `web_client/vite.config.ts`:
  ```typescript
  server: {
    allowedHosts: ['.ngrok-free.dev'],
  }
  ```

---

## 📊 Parametri di Training GPU

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `WARMUP_SIZE` | 50.000 | Esperienze prima di iniziare il training |
| `TARGET_TRAIN_RATIO` | 8 | Rapporto env steps / training steps |
| `TARGET_UPDATE_ENV_STEPS` | 10.000 | Frequenza aggiornamento target network |
| `WEIGHT_SYNC_INTERVAL` | 500 | Sync pesi al main thread ogni N train steps |
| `TRAIN_TIME_BUDGET_MS` | 25ms | Budget tempo per backprop per loop |

### Batch Size Automatico

Il batch size è auto-regolato in base al numero di bird:

| Birds | Batch Size |
|-------|------------|
| < 100 | 64 |
| 100-499 | 128 |
| 500-1999 | 256 |
| 2000+ | 512 |

---

## 🔄 Epsilon Decay

Il decay di epsilon è stato **mantenuto lineare** come nella fork originale, basato sui **training steps**:

```typescript
function updateEpsilon(): void {
  const stepsSinceDecayStart = trainCallCount - decayStartTrainStep
  const frac = Math.min(1.0, stepsSinceDecayStart / config.epsilonDecaySteps)
  epsilon = decayStartEpsilon + frac * (config.epsilonEnd - decayStartEpsilon)
}
```

Parametri default:
- `epsilonStart`: 0.5
- `epsilonEnd`: 0.05
- `epsilonDecaySteps`: 150.000 training steps

---

## 🛠️ Come Usare le Nuove Funzionalità

### Attivare GPU Training

1. Nella pagina principale, clicca "Train AI"
2. Seleziona la checkbox "🎮 GPU Training"
3. Regola il numero di bird paralleli con lo slider
4. Attiva "⚡ Fast Training" per massima velocità
5. Clicca "Start Training"

### Nota su WebGPU vs WebGL

- **WebGPU**: Massime prestazioni, richiede Chrome/Edge recente con flag abilitati
- **WebGL**: Fallback automatico, funziona su tutti i browser moderni

---

## 📁 Struttura File Aggiunti

```
web_client/src/rl/
├── NeuralNetworkTF.ts    # Rete neurale TensorFlow.js
├── GPUDQNAgent.ts        # Agente DQN per GPU
├── GPUTrainingLoop.ts    # Training loop GPU
└── gpu.worker.ts         # Web Worker multi-bird
```

---

## 🎯 Roadmap Future

- [ ] Double DQN per ridurre overestimation
- [ ] Dueling DQN per migliore value estimation
- [ ] Prioritized Experience Replay
- [ ] Mobile touch controls optimization

---

**Autore:** Paolo  
**Data:** Dicembre 2024  
**Fork originale:** [giosilvi/flappy_ai](https://github.com/giosilvi/flappy_ai)

