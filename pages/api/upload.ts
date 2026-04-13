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

    // Fix 1: Cast the result to 'any' to allow destructuring [dataset]
    const [dataset] = (await sql`
      INSERT INTO datasets (name, row_count)
      VALUES (${filename ?? 'upload'}, ${events.length})
      RETURNING id, name, uploaded_at, row_count
    `) as any;

    const CHUNK_SIZE = 500;
    for (let i = 0; i < events.length; i += CHUNK_SIZE) {
      const chunk = events.slice(i, i + CHUNK_SIZE);
      
      const rowsToInsert = chunk.map(e => ({
        dataset_id: dataset.id,
        invariant_mass: e.invariant_mass,
        particle_type: e.particle_type,
        combination: e.combination
      }));

      // Fix 2: Cast the helper function CALL to 'any'
      // This tells TS: "Don't validate this as a TemplateStringsArray"
      await sql`
        INSERT INTO events 
        ${(sql as any)(rowsToInsert, 'dataset_id', 'invariant_mass', 'particle_type', 'combination')}
      `;
    }

    return res.status(200).json({ dataset });
  } catch (err: any) {
    console.error('Upload error:', err);
    return res.status(500).json({ error: err.message });
  }
}