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
    if (events.length === 0) return res.status(400).json({ error: 'No valid data rows found in file' });

    const sql = getDb();

    // Insert dataset record
    const [dataset] = await sql`
      INSERT INTO datasets (name, row_count)
      VALUES (${filename ?? 'upload'}, ${events.length})
      RETURNING id, name, uploaded_at, row_count
    `;

    // Bulk insert events in chunks of 500
    const CHUNK = 500;
    for (let i = 0; i < events.length; i += CHUNK) {
      const chunk = events.slice(i, i + CHUNK);
      // Build parameterised bulk insert using neon tagged template
      for (const e of chunk) {
        await sql`
          INSERT INTO events (dataset_id, invariant_mass, particle_type, combination)
          VALUES (${dataset.id}, ${e.invariant_mass}, ${e.particle_type}, ${e.combination})
        `;
      }
    }

    return res.status(200).json({ dataset });
  } catch (err: any) {
    console.error(err);
    return res.status(500).json({ error: err.message ?? 'Internal server error' });
  }
}
