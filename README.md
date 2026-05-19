# ChatCommissioning — AI-Assisted Drive Commissioning via MCP

This repository accompanies the paper:

> **"Chat with Your Drive: MCP Server Allowing Safe and Efficient Drive Commissioning by AI Agents"**
> V. Šmidl, Z. Peroutka — ICEM 2026

It provides an MCP server that exposes an IPMSM drive simulator to an AI agent (Claude), enabling a guided commissioning workflow entirely through natural language.

## What this does

The MCP server wraps an IPMSM simulator and FCS-MPC / PI controller and exposes 18 tools and 4 resources to any MCP-compatible AI agent. The agent can:

- Read motor parameters and the drive capabilities manifest
- Run resistance identification and flux-linkage map sweeps
- Set MTPA or field-weakening current references
- Tune FCS-MPC cost weights and PI speed controller gains
- Execute step-response and load-profile scenarios
- Receive inline plots directly in the chat

A `CLAUDE.md` instruction file governs the interaction style: the agent operates as a guided co-pilot, pausing after each step and explaining its reasoning before asking for confirmation.

## Architecture

```
Engineer (natural language)
    ↓
AI Agent  (Claude — control theory knowledge)
    ↓  MCP tools / resources
MCP Server  (server.py — safety boundary, i ≤ i_max)
    ↓
IPMSM Simulator + FCS-MPC / PI Controller
```

## Requirements

- Python 3.12
- [Claude Code](https://claude.ai/code) CLI (`npm install -g @anthropic-ai/claude-code`)
- An Anthropic API key (`export ANTHROPIC_API_KEY=sk-ant-...`)

## Installation

```bash
git clone https://github.com/AI-RICE/ChatCommissioning.git
cd ChatCommissioning

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Verify the simulator works:

```bash
.venv/bin/python validate_simulator.py
```

**Windows:** replace `.venv/bin/python` with `.venv\Scripts\python.exe` throughout, and update `.mcp.json` accordingly.

## Running the commissioning session

Start Claude Code from the repository root:

```bash
claude
```

Claude Code automatically reads `CLAUDE.md` and `.mcp.json`. The server starts on first tool call. To begin a commissioning session, type:

```
Commission the IPMSM drive for a 7 A operating point.
```

The agent will read the capabilities manifest, summarise motor parameters, and then follow the guided commissioning sequence defined in `CLAUDE.md`, pausing for your confirmation at each step.

## File overview

| File / folder | Purpose |
|---|---|
| `server.py` | MCP server — 18 tools, 4 resources, safety constraints |
| `simulator/ipmsm.py` | IPMSM plant model in the dq frame with saturation |
| `controller/fcs_mpc.py` | FCS-MPC and PI controller implementations |
| `CLAUDE.md` | Agent instruction file — interaction mode, commissioning sequence, embedded control-theory knowledge |
| `.mcp.json` | MCP server registration (relative paths, edit for Windows) |
| `.claude/settings.local.json` | Pre-approved tool permissions (avoids per-call prompts) |
| `validate_simulator.py` | Quick sanity check of the simulator |

## Commissioning sequence

The agent follows this sequence by default (each step pauses for confirmation):

1. Read motor parameters from `drive://capabilities`
2. Identify stator resistance (`identify_resistance`)
3. Sweep flux-linkage map (`sweep_flux_linkage_map`, `plot_flux_linkage_map`)
4. Set MTPA current reference (`set_mtpa_reference`)
5. Run baseline scenario and evaluate FCS-MPC ripple (`run_scenario`, `plot_results`)
6. Tune cost weights (`set_controller_weights`) and verify
7. Run load profile and evaluate PI speed-loop gains (`run_load_profile`)
8. Present commissioning report

## Safety

Current limits (`i ≤ i_max`) are enforced as soft rejects in the MCP server tools. Voltage clamping is applied in the simulator before every integration step. The agent cannot bypass these constraints — all drive interactions go through the declared MCP interface.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{smidl2026chat,
  author    = {Šmidl, Václav and Peroutka, Zdeněk},
  title     = {Chat with Your Drive: {MCP} Server Allowing Safe and Efficient
               Drive Commissioning by {AI} Agents},
  booktitle = {Proc. Int. Conf. Electrical Machines (ICEM)},
  year      = {2026}
}
```
