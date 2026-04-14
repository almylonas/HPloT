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

    // 2. Bulk insert in chunks using objects (required by Neon/Postgres sql() helper)
    const CHUNK_SIZE = 500;
    const COLUMNS = ['dataset_id', 'invariant_mass', 'particle_type', 'combination'] as const;

    for (let i = 0; i < events.length; i += CHUNK_SIZE) {
      const chunk = events.slice(i, i + CHUNK_SIZE);

      const values = chunk.map(e => ({
        dataset_id: dataset.id,
        invariant_mass: e.invariant_mass,
        particle_type: e.particle_type,
        combination: e.combination,
      }));

      await sql`
        INSERT INTO events (dataset_id, invariant_mass, particle_type, combination)
        VALUES ${(sql as any)(values, COLUMNS)}
      `;
    }

    return res.status(200).json({ dataset });
  } catch (err: any) {
    console.error('Database Error:', err);
    return res.status(500).json({ error: err.message });
  }
}