import type { NextApiRequest, NextApiResponse } from 'next';
import { getDb } from '../../lib/db';
import { parseFileContent } from '../../lib/parser';
 
export const config = {
  api: { bodyParser: { sizeLimit: '10mb' } },
};
 
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
 
  try {
    const { filename, content } = req.body as { filename: string; content: string };
    if (!content) return res.status(400).json({ error: 'No content provided' });
 
    const events = parseFileContent(content);
    if (events.length === 0) return res.status(400).json({ error: 'No valid data rows found' });
 
    const sql = getDb();
 
    // 1. Insert dataset
    const [dataset] = (await sql`
      INSERT INTO datasets (name, row_count)
      VALUES (${filename ?? 'upload'}, ${events.length})
      RETURNING id
    `) as any;
 
    // 2. Bulk insert in chunks.
    // neon() returns a tagged-template function. Calling it as a regular function
    // with (queryString, paramsArray) is the correct way to run a raw parameterized
    // query — no .query() method exists on this driver.
    const CHUNK_SIZE = 500;
 
    for (let i = 0; i < events.length; i += CHUNK_SIZE) {
      const chunk = events.slice(i, i + CHUNK_SIZE);
 
      // Build: ($1,$2,$3,$4), ($5,$6,$7,$8), ...
      const placeholders = chunk
        .map((_, idx) => {
          const base = idx * 4;
          return `($${base + 1}, $${base + 2}, $${base + 3}, $${base + 4})`;
        })
        .join(', ');
 
      const values = chunk.flatMap(e => [
        dataset.id,
        e.invariant_mass,
        e.particle_type,
        e.combination,
      ]);
 
      await (sql as any)(
        `INSERT INTO events (dataset_id, invariant_mass, particle_type, combination) VALUES ${placeholders}`,
        values,
      );
    }
 
    return res.status(200).json({ dataset });
  } catch (err: any) {
    console.error('Database Error:', err);
    return res.status(500).json({ error: err.message });
  }
}