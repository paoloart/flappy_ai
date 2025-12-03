/**
 * Simple Leaderboard API Server
 * Stores leaderboard in a local JSON file
 */

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

// Data file path - persisted in a volume
const DATA_DIR = process.env.DATA_DIR || './data';
const LEADERBOARD_FILE = path.join(DATA_DIR, 'leaderboard.json');

// Middleware
app.use(cors());
app.use(express.json());

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Initialize leaderboard file if it doesn't exist
function getLeaderboard() {
  try {
    if (fs.existsSync(LEADERBOARD_FILE)) {
      const data = fs.readFileSync(LEADERBOARD_FILE, 'utf-8');
      return JSON.parse(data);
    }
  } catch (error) {
    console.error('Error reading leaderboard:', error);
  }
  return { entries: [] };
}

function saveLeaderboard(data) {
  try {
    fs.writeFileSync(LEADERBOARD_FILE, JSON.stringify(data, null, 2));
  } catch (error) {
    console.error('Error saving leaderboard:', error);
    throw error;
  }
}

// GET /api/leaderboard - Get leaderboard entries
app.get('/api/leaderboard', (req, res) => {
  const limit = parseInt(req.query.limit) || 10;
  const data = getLeaderboard();
  
  // Sort by score descending
  const sortedEntries = [...data.entries].sort((a, b) => b.score - a.score);
  
  // Limit entries
  const limitedEntries = sortedEntries.slice(0, limit);
  
  // Mark champion
  limitedEntries.forEach((entry, index) => {
    entry.isChampion = index === 0;
  });
  
  const champion = limitedEntries.length > 0 ? limitedEntries[0] : null;
  
  res.json({
    entries: limitedEntries,
    champion,
  });
});

// POST /api/leaderboard - Submit a new score
app.post('/api/leaderboard', (req, res) => {
  const { name, pipes, params, architecture, score } = req.body;
  
  if (!name || typeof pipes !== 'number' || typeof params !== 'number') {
    return res.status(400).json({ error: 'Missing required fields: name, pipes, params' });
  }
  
  const data = getLeaderboard();
  
  // Create new entry
  const newEntry = {
    id: `entry-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    name: name.trim().substring(0, 20), // Limit name length
    score: score, // Pre-calculated adjusted score from client
    pipes,
    params,
    architecture: architecture || 'unknown',
    createdAt: new Date().toISOString(),
  };
  
  // Check if this is a new champion
  const sortedEntries = [...data.entries].sort((a, b) => b.score - a.score);
  const currentChampion = sortedEntries[0];
  const isNewChampion = !currentChampion || score > currentChampion.score;
  
  // Add new entry
  data.entries.push(newEntry);
  
  // Save
  saveLeaderboard(data);
  
  console.log(`[Leaderboard] New entry: ${newEntry.name} - ${newEntry.score} pts (${pipes} pipes)`);
  
  res.json({
    success: true,
    entry: newEntry,
    isNewChampion,
  });
});

// GET /api/leaderboard/lowest - Get the lowest score threshold
app.get('/api/leaderboard/lowest', (req, res) => {
  const data = getLeaderboard();
  const sortedEntries = [...data.entries].sort((a, b) => b.score - a.score);
  
  // If less than 10 entries, threshold is 0 (anyone can join)
  if (sortedEntries.length < 10) {
    return res.json({ lowestScore: 0 });
  }
  
  // Otherwise return the 10th entry's score
  res.json({ lowestScore: sortedEntries[9]?.score || 0 });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`🏆 Leaderboard API running on port ${PORT}`);
  console.log(`   Data stored in: ${LEADERBOARD_FILE}`);
});

