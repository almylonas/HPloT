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

    // 1. Insert dataset (This part works!)
    const [dataset] = (await sql`
      INSERT INTO datasets (name, row_count)
      VALUES (${filename ?? 'upload'}, ${events.length})
      RETURNING id
    `) as any;

    // 2. Optimized Bulk Insert using standard array mapping
    // This avoids the sql() helper and the $1 syntax error entirely.
    const CHUNK_SIZE = 500;
    for (let i = 0; i < events.length; i += CHUNK_SIZE) {
      const chunk = events.slice(i, i + CHUNK_SIZE);
      
      // We map the chunk to a simple array of values for each row
      const values = chunk.map(e => [
        dataset.id,
        e.invariant_mass,
        e.particle_type,
        e.combination
      ]);

      // Using the VALUES (...) syntax with the values array
      // is the most compatible way to handle this in Postgres.
      await sql`
        INSERT INTO events (dataset_id, invariant_mass, particle_type, combination)
        VALUES ${sql(values)}
      `;
    }

    return res.status(200).json({ dataset });
  } catch (err: any) {
    console.error('Database Error:', err);
    return res.status(500).json({ error: err.message });
  }
}