// @ts-check

/**
 * A lifecycle written down as data: for each state, the events it takes and the state each
 * leads to. An event a state does not list is not a move from it. The module that owns a
 * lifecycle declares it here, beside the one function that makes its moves, and nothing else
 * writes the state.
 *
 * @template {string} State
 * @template {string} Event
 * @typedef {Object} Machine
 * @property {string} name
 * @property {State} initial
 * @property {Readonly<Record<State, Readonly<Partial<Record<Event, State>>>>>} moves
 * @property {readonly State[]} states
 * @property {readonly Event[]} events every event some state takes
 * @property {(from:State, event:Event)=>State|null} next the state `event` leads to from
 *   `from`, or null where `from` does not take it
 */

/**
 * Declare a lifecycle. A table that cannot be right is refused where it is declared: a move to
 * a state the table does not define, a state no move from `initial` reaches, and a state no
 * move leaves, which would hold whatever enters it for good.
 *
 * @template {string} State
 * @template {string} Event
 * @param {{name:string, initial:State, moves:Record<State, Partial<Record<Event, State>>>}} table
 * @returns {Machine<State, Event>}
 */
export function defineMachine({ name, initial, moves }) {
  const states = /** @type {State[]} */ (Object.keys(moves));
  /** @param {State} state @returns {[Event, State][]} */
  const movesFrom = (state) => /** @type {[Event, State][]} */ (Object.entries(moves[state]));
  /** @type {Set<Event>} */
  const events = new Set();
  for (const state of states) {
    for (const [event, target] of movesFrom(state)) {
      events.add(event);
      if (!states.includes(target)) {
        throw new Error(`${name}: ${state} --${event}--> ${target}, which is not a state.`);
      }
    }
    if (!movesFrom(state).some(([, target]) => target !== state)) {
      throw new Error(`${name}: no move leaves ${state}.`);
    }
  }
  const reached = new Set([initial]);
  for (const state of reached) {
    for (const [, target] of movesFrom(state)) reached.add(target);
  }
  const unreached = states.filter((state) => !reached.has(state));
  if (unreached.length) {
    throw new Error(`${name}: no move from ${initial} reaches ${unreached.join(", ")}.`);
  }
  const table = /** @type {Record<State, Readonly<Partial<Record<Event, State>>>>} */ ({});
  for (const state of states) table[state] = Object.freeze({ ...moves[state] });
  return Object.freeze({
    name,
    initial,
    moves: Object.freeze(table),
    states: Object.freeze(states),
    events: Object.freeze([...events]),
    next: (from, event) => table[from]?.[event] ?? null
  });
}
