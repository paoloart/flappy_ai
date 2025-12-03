/**
 * Tests for the Leaderboard API
 * Run with: npm test
 */

const assert = require('assert');
const http = require('http');

const API_BASE = process.env.API_URL || 'http://localhost:3001';

// Helper to make HTTP requests
function request(method, path, body = null) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, API_BASE);
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname + url.search,
      method,
      headers: {
        'Content-Type': 'application/json',
      },
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode, data });
        }
      });
    });

    req.on('error', reject);
    
    if (body) {
      req.write(JSON.stringify(body));
    }
    req.end();
  });
}

async function runTests() {
  console.log('🧪 Running Leaderboard API Tests...\n');
  let passed = 0;
  let failed = 0;

  // Test 1: Health check
  try {
    const res = await request('GET', '/api/health');
    assert.strictEqual(res.status, 200, 'Health check should return 200');
    assert.strictEqual(res.data.status, 'ok', 'Health check should return status ok');
    console.log('✅ Test 1: Health check passed');
    passed++;
  } catch (e) {
    console.log('❌ Test 1: Health check failed -', e.message);
    failed++;
  }

  // Test 2: Get empty leaderboard
  try {
    const res = await request('GET', '/api/leaderboard?limit=10');
    assert.strictEqual(res.status, 200, 'Get leaderboard should return 200');
    assert(Array.isArray(res.data.entries), 'Entries should be an array');
    console.log('✅ Test 2: Get leaderboard passed');
    passed++;
  } catch (e) {
    console.log('❌ Test 2: Get leaderboard failed -', e.message);
    failed++;
  }

  // Test 3: Submit a score
  const testEntry = {
    name: 'TestBot',
    pipes: 10,
    params: 8706,
    architecture: '6→64→64→2',
    score: 10.0,
  };
  
  try {
    const res = await request('POST', '/api/leaderboard', testEntry);
    assert.strictEqual(res.status, 200, 'Submit should return 200');
    assert.strictEqual(res.data.success, true, 'Submit should succeed');
    assert(res.data.entry, 'Should return the entry');
    assert.strictEqual(res.data.entry.name, 'TestBot', 'Entry name should match');
    assert.strictEqual(res.data.entry.pipes, 10, 'Entry pipes should match');
    console.log('✅ Test 3: Submit score passed');
    passed++;
  } catch (e) {
    console.log('❌ Test 3: Submit score failed -', e.message);
    failed++;
  }

  // Test 4: Verify entry appears in leaderboard
  try {
    const res = await request('GET', '/api/leaderboard?limit=10');
    assert.strictEqual(res.status, 200, 'Get leaderboard should return 200');
    const testBotEntry = res.data.entries.find(e => e.name === 'TestBot');
    assert(testBotEntry, 'TestBot should be in leaderboard');
    console.log('✅ Test 4: Entry appears in leaderboard passed');
    passed++;
  } catch (e) {
    console.log('❌ Test 4: Entry appears in leaderboard failed -', e.message);
    failed++;
  }

  // Test 5: Submit higher score becomes champion
  const betterEntry = {
    name: 'ChampionBot',
    pipes: 50,
    params: 8706,
    architecture: '6→64→64→2',
    score: 50.0,
  };
  
  try {
    const res = await request('POST', '/api/leaderboard', betterEntry);
    assert.strictEqual(res.status, 200, 'Submit should return 200');
    assert.strictEqual(res.data.isNewChampion, true, 'Should be new champion');
    console.log('✅ Test 5: New champion detection passed');
    passed++;
  } catch (e) {
    console.log('❌ Test 5: New champion detection failed -', e.message);
    failed++;
  }

  // Test 6: Get lowest score threshold
  try {
    const res = await request('GET', '/api/leaderboard/lowest');
    assert.strictEqual(res.status, 200, 'Get lowest should return 200');
    assert(typeof res.data.lowestScore === 'number', 'Should return a number');
    console.log('✅ Test 6: Get lowest score passed');
    passed++;
  } catch (e) {
    console.log('❌ Test 6: Get lowest score failed -', e.message);
    failed++;
  }

  // Test 7: Validate required fields
  try {
    const res = await request('POST', '/api/leaderboard', { name: 'NoData' });
    assert.strictEqual(res.status, 400, 'Should return 400 for missing fields');
    console.log('✅ Test 7: Validation passed');
    passed++;
  } catch (e) {
    console.log('❌ Test 7: Validation failed -', e.message);
    failed++;
  }

  // Test 8: Name length limit
  try {
    const longNameEntry = {
      name: 'ThisIsAVeryLongNameThatShouldBeTruncated',
      pipes: 5,
      params: 8706,
      architecture: '6→64→64→2',
      score: 5.0,
    };
    const res = await request('POST', '/api/leaderboard', longNameEntry);
    assert.strictEqual(res.status, 200, 'Submit should return 200');
    assert(res.data.entry.name.length <= 20, 'Name should be truncated to 20 chars');
    console.log('✅ Test 8: Name length limit passed');
    passed++;
  } catch (e) {
    console.log('❌ Test 8: Name length limit failed -', e.message);
    failed++;
  }

  // Summary
  console.log('\n' + '═'.repeat(50));
  console.log(`📊 Results: ${passed} passed, ${failed} failed`);
  console.log('═'.repeat(50));

  if (failed > 0) {
    process.exit(1);
  }
}

// Run tests
runTests().catch(err => {
  console.error('❌ Test runner error:', err);
  process.exit(1);
});

