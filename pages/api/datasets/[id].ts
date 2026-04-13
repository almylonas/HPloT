import type { NextApiRequest, NextApiResponse } from 'next';
// Changed from '../../lib/db' to '../../../lib/db'
import { getDb } from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'DELETE') {
    res.setHeader('Allow', ['DELETE']);
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { id } = req.query;
  const numericId = Number(id);

  if (!id || isNaN(numericId)) {
    return res.status(400).json({ error: 'Valid ID is required' });
  }

  try {
    const sql = getDb();

    const result = await sql`
      DELETE FROM datasets 
      WHERE id = ${numericId}
      RETURNING id
    `;

    if (result.length === 0) {
      return res.status(404).json({ error: 'Dataset not found' });
    }

    return res.status(200).json({ ok: true });
  } catch (err: any) {
    console.error('Delete error:', err);
    return res.status(500).json({ error: err.message });
  }
}