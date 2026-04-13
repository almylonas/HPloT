import type { NextApiRequest, NextApiResponse } from 'next';
import { getDb } from '../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // 1. Method Guard
  if (req.method !== 'DELETE') {
    res.setHeader('Allow', ['DELETE']);
    return res.status(405).json({ error: `Method ${req.method} not allowed` });
  }

  const { id } = req.query;

  // 2. Validation: Ensure ID exists and is a valid number
  const datasetId = Number(id);
  if (!id || isNaN(datasetId)) {
    return res.status(400).json({ error: 'Invalid or missing ID' });
  }

  try {
    const sql = getDb();

    // 3. Perform Deletion
    // Using a template literal with the postgres library safely handles parameterization
    const result = await sql`
      DELETE FROM datasets 
      WHERE id = ${datasetId}
      RETURNING id
    `;

    // 4. Handle Case where ID doesn't exist
    if (result.length === 0) {
      return res.status(404).json({ error: 'Dataset not found' });
    }

    return res.status(200).json({ ok: true, deletedId: datasetId });
  } catch (err: any) {
    console.error('Database error:', err);
    return res.status(500).json({ 
      error: 'Failed to delete dataset', 
      message: err.message 
    });
  }
}