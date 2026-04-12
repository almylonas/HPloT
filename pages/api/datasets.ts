import type { NextApiRequest, NextApiResponse } from 'next';
import { getDb } from '../../lib/db';

export default async function handler(_req: NextApiRequest, res: NextApiResponse) {
  try {
    const sql = getDb();
    const datasets = await sql`
      SELECT id, name, uploaded_at, row_count, description
      FROM datasets
      ORDER BY uploaded_at DESC
    `;
    return res.status(200).json({ datasets });
  } catch (err: any) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
}
