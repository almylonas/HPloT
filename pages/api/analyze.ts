import type { NextApiRequest, NextApiResponse } from 'next';
import { getDb } from '../../lib/db';

// Define the shape of your database row for type safety
interface EventRow {
  invariant_mass: number;
  particle_type: string;
  combination: string;
}

const ENERGY_RANGES: Record<string, [number, number]> = {
  R1: [2, 4],
  R2: [7, 13],
  R3: [80, 100],
  R4: [900, 1100],
  R5: [1400, 1600],
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { datasetIds, numBins = 50 } = req.body as {
      datasetIds: number[] | 'all';
      numBins: number;
    };

    const sql = getDb();
    const isAll = datasetIds === 'all' || !datasetIds;
    const ids: number[] = isAll ? [] : (datasetIds as number[]);

    // Fixed: Initialize as an empty array of EventRow
    let rows: EventRow[] = [];

    if (isAll) {
      // Use "as unknown as EventRow[]" to bypass the Result object type mismatch
      rows = (await sql`
        SELECT invariant_mass, particle_type, combination FROM events
      `) as unknown as EventRow[];
    } else if (ids.length > 0) {
      rows = (await sql`
        SELECT invariant_mass, particle_type, combination FROM events
        WHERE dataset_id = ANY(${ids})
      `) as unknown as EventRow[];
    } else {
      return res.status(200).json({ bins: {}, stats: {}, totalEvents: 0 });
    }

    // ------ Histogram bins computation ------
    function buildBins(values: number[], nBins: number) {
      if (values.length === 0) return { x: [], y: [] };
      const min = Math.min(...values);
      const max = Math.max(...values);
      
      if (min === max) return { x: [min], y: [values.length] };
      
      const width = (max - min) / nBins;
      const counts = new Array(nBins).fill(0);
      
      for (const v of values) {
        let idx = Math.floor((v - min) / width);
        if (idx >= nBins) idx = nBins - 1;
        counts[idx]++;
      }
      
      const x = Array.from({ length: nBins }, (_, i) => +(min + (i + 0.5) * width).toFixed(4));
      return { x, y: counts };
    }

    // Helper to extract mass by combination type
    const getMass = (type: string) => rows.filter(r => r.combination === type).map(r => r.invariant_mass);

    const bins = {
      dileptonE: buildBins(getMass('e'), numBins),
      dileptonM: buildBins(getMass('m'), numBins),
      fourEE: buildBins(getMass('4ee'), numBins),
      fourMM: buildBins(getMass('4mm'), numBins),
      fourME: buildBins(getMass('4me'), numBins),
      diphoton: buildBins(getMass('g'), numBins),
    };

    // ------ Statistics ------
    function rangeStats(values: number[]) {
      return Object.entries(ENERGY_RANGES).map(([name, [lo, hi]]) => {
        const filtered = values.filter((v) => v >= lo && v <= hi);
        const mean = filtered.length > 0 
          ? filtered.reduce((a, b) => a + b, 0) / filtered.length 
          : null;
          
        return { 
          range: `${name} (${lo}–${hi} GeV)`, 
          events: filtered.length, 
          mean: mean !== null ? +mean.toFixed(2) : 'N/A' 
        };
      });
    }

    const stats = {
      electrons: rangeStats(getMass('e')),
      muons: rangeStats(getMass('m')),
      photons: rangeStats(getMass('g')),
    };

    return res.status(200).json({ bins, stats, totalEvents: rows.length });
  } catch (err: any) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
}