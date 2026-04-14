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

    // 1. Insert dataset (Cast to any to allow destructuring)
    const [dataset] = (await sql`
      INSERT INTO datasets (name, row_count)
      VALUES (${filename ?? 'upload'}, ${events.length})
      RETURNING id
    `) as any;

    // 2. Bulk Insert with fixed Type-Safe syntax
    const CHUNK_SIZE = 500;
    for (let i = 0; i < events.length; i += CHUNK_SIZE) {
      const chunk = events.slice(i, i + CHUNK_SIZE);
      
      const values = chunk.map(e => [
        dataset.id,
        e.invariant_mass,
        e.particle_type,
        e.combination
      ]);

      // THE FIX: 
      // Instead of nesting sql inside ${}, we use the helper as a standalone 
      // and cast it to 'any' to satisfy the strict Neon/Postgres TemplateStringsArray check.
      await sql`
        INSERT INTO events (dataset_id, invariant_mass, particle_type, combination)
        VALUES ${(sql as any)(values)}
      `;
    }

    return res.status(200).json({ dataset });
  } catch (err: any) {
    console.error('Database Error:', err);
    return res.status(500).json({ error: err.message });
  }
}