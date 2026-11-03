// @ts-check

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {{key:string, label:string, role:"average"|"spread", terms:string[]}} ParameterGroup */

/**
 * Whether the payload is a distributional model's: its terms carry the
 * family parameter they act in. A SuperGLM payload carries none.
 * @param {Record<string, TermPayload>} terms
 */
export function hasParameters(terms) {
  return Object.values(terms).some((term) => term.parameter_role !== undefined);
}

/**
 * The terms grouped by family parameter, in payload order with each
 * parameter's first appearance first. A feature acting in two parameters is
 * a term in each group. A term with no parameter fields belongs to no group.
 * @param {Record<string, TermPayload>} terms
 * @returns {ParameterGroup[]}
 */
export function parameterGroups(terms) {
  /** @type {Map<string, ParameterGroup>} */
  const groups = new Map();
  for (const [name, term] of Object.entries(terms)) {
    const { parameter, parameter_label: label, parameter_role: role } = term;
    if (parameter === undefined || label === undefined || (role !== "average" && role !== "spread")) {
      continue;
    }
    let group = groups.get(parameter);
    if (!group) {
      group = { key: parameter, label, role, terms: [] };
      groups.set(parameter, group);
    }
    group.terms.push(name);
  }
  return [...groups.values()];
}
