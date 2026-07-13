// @ts-check

import { HELP_SECTIONS, OPERATION_HELP, TOOL_HELP } from "./help_content.js";

/**
 * Render the shared editor help catalog into the inspector Help pane.
 *
 * @param {HTMLElement} root
 */
export function renderHelpDrawer(root) {
  root.replaceChildren(...HELP_SECTIONS.map(sectionNode));
}

/** @param {import('./help_content.js').HelpSection} section @returns {HTMLElement} */
function sectionNode(section) {
  const sectionElement = document.createElement("section");
  sectionElement.className = "help-section";

  const heading = document.createElement("h3");
  heading.textContent = section.title;
  sectionElement.append(heading);

  const list = document.createElement("dl");
  for (const key of section.keys || []) {
    const entry = TOOL_HELP[key] || OPERATION_HELP[key];
    if (!entry) continue;
    const term = document.createElement("dt");
    term.textContent = entry.shortcut ? `${entry.title} (${entry.shortcut})` : entry.title;
    const description = document.createElement("dd");
    description.textContent = entry.body;
    list.append(term, description);
  }
  for (const item of section.items || []) {
    const term = document.createElement("dt");
    term.textContent = item;
    list.append(term);
  }
  sectionElement.append(list);
  return sectionElement;
}
