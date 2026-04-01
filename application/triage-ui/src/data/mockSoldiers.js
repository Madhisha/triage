/**
 * Mock Soldier Data & Vital Signs Simulator
 * 
 * Generates 6 soldier profiles with baseline vitals. 
 * simulateVitals() adds realistic random drift each tick.
 */

const SOLDIERS = [
  {
    id: 'SOL-001',
    name: 'Sgt. Marcus Rivera',
    callsign: 'VIPER-1',
    rank: 'Sergeant',
    unit: '3rd Infantry Division',
    bloodType: 'O+',
    age: 28,
    missionStatus: 'Active Patrol',
    chiefcomplaint: 'blast exposure, headache',
    baseline: { temperature: 98.8, heartrate: 82, resprate: 17, o2sat: 97, sbp: 128, dbp: 82, pain: 3 },
  },
  {
    id: 'SOL-002',
    name: 'Cpl. Aisha Thompson',
    callsign: 'HAWK-3',
    rank: 'Corporal',
    unit: '82nd Airborne',
    bloodType: 'A+',
    age: 25,
    missionStatus: 'Forward Recon',
    chiefcomplaint: 'dehydration, dizziness',
    baseline: { temperature: 99.5, heartrate: 95, resprate: 20, o2sat: 95, sbp: 110, dbp: 70, pain: 4 },
  },
  {
    id: 'SOL-003',
    name: 'Pvt. James Kowalski',
    callsign: 'GHOST-7',
    rank: 'Private',
    unit: '10th Mountain Division',
    bloodType: 'B-',
    age: 22,
    missionStatus: 'Base Defense',
    chiefcomplaint: 'gunshot wound to right leg',
    baseline: { temperature: 100.2, heartrate: 115, resprate: 24, o2sat: 93, sbp: 95, dbp: 60, pain: 8 },
  },
  {
    id: 'SOL-004',
    name: 'SSgt. Elena Vasquez',
    callsign: 'STORM-2',
    rank: 'Staff Sergeant',
    unit: '1st Special Forces Group',
    bloodType: 'AB+',
    age: 31,
    missionStatus: 'Convoy Escort',
    chiefcomplaint: 'none',
    baseline: { temperature: 98.4, heartrate: 68, resprate: 14, o2sat: 99, sbp: 122, dbp: 78, pain: 0 },
  },
  {
    id: 'SOL-005',
    name: 'Spc. David Chen',
    callsign: 'RAPTOR-5',
    rank: 'Specialist',
    unit: '75th Ranger Regiment',
    bloodType: 'O-',
    age: 26,
    missionStatus: 'Overwatch Position',
    chiefcomplaint: 'chest tightness, shortness of breath',
    baseline: { temperature: 99.1, heartrate: 105, resprate: 22, o2sat: 94, sbp: 140, dbp: 90, pain: 6 },
  },
  {
    id: 'SOL-006',
    name: 'Pfc. Sarah Mitchell',
    callsign: 'ECHO-4',
    rank: 'Private First Class',
    unit: '101st Airborne',
    bloodType: 'A-',
    age: 23,
    missionStatus: 'Medevac Standby',
    chiefcomplaint: 'heat exhaustion, nausea',
    baseline: { temperature: 101.0, heartrate: 108, resprate: 21, o2sat: 96, sbp: 105, dbp: 65, pain: 5 },
  },
]

// Clamp value within valid clinical ranges
function clamp(val, min, max) {
  return Math.min(max, Math.max(min, val))
}

// Add random gaussian-like noise
function jitter(value, stddev) {
  // Box-Muller transform for approximate normal distribution
  const u1 = Math.random()
  const u2 = Math.random()
  const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  return value + normal * stddev
}

/**
 * Simulate realistic vital sign changes for a soldier.
 * Takes current vitals and returns new vitals with random drift.
 */
export function simulateVitals(currentVitals, baseline) {
  // Pull slightly toward baseline (mean reversion) + random drift
  const reversion = 0.15 // how strongly we pull back to baseline
  const drift = (current, base, noise, min, max) => {
    const pulled = current + (base - current) * reversion
    return clamp(Math.round(jitter(pulled, noise) * 10) / 10, min, max)
  }

  return {
    temperature:  drift(currentVitals.temperature, baseline.temperature, 0.2, 95.0, 105.0),
    heartrate:    Math.round(drift(currentVitals.heartrate, baseline.heartrate, 3, 40, 180)),
    resprate:     Math.round(drift(currentVitals.resprate, baseline.resprate, 1.5, 6, 45)),
    o2sat:        Math.round(drift(currentVitals.o2sat, baseline.o2sat, 1, 70, 100)),
    sbp:          Math.round(drift(currentVitals.sbp, baseline.sbp, 4, 60, 220)),
    dbp:          Math.round(drift(currentVitals.dbp, baseline.dbp, 3, 35, 140)),
    pain:         Math.round(clamp(drift(currentVitals.pain, baseline.pain, 0.5, 0, 10), 0, 10)),
  }
}

/**
 * Initialize all soldiers with their baseline vitals
 */
export function initializeSoldiers() {
  return SOLDIERS.map(soldier => ({
    ...soldier,
    vitals: { ...soldier.baseline },
    prediction: null,
    history: [],       // last N vitals snapshots
    lastUpdated: null,
  }))
}

export default SOLDIERS
