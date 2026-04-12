export interface ParsedEvent {
  invariant_mass: number;
  particle_type: number;
  combination: string;
}

/**
 * Parses CSV/TXT content in the format:
 *   invariant_mass,particle_type,combination
 * Lines with missing or non-numeric mass/particle are silently skipped.
 */
export function parseFileContent(text: string): ParsedEvent[] {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  const events: ParsedEvent[] = [];

  for (const line of lines) {
    // Support comma or tab as separator
    const parts = line.split(/,|\t/).map((p) => p.trim());

    if (parts.length < 2) continue;

    const mass = parseFloat(parts[0]);
    const ptype = parseInt(parts[1], 10);

    if (isNaN(mass) || isNaN(ptype)) continue;

    const combination = (parts[2] ?? '').toLowerCase().trim();

    events.push({ invariant_mass: mass, particle_type: ptype, combination });
  }

  return events;
}
