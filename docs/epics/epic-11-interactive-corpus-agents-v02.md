# Epic 11: Interactive Corpus Agents

**Epic ID:** E11-v0.2
**Owner:** Andy
**Status:** Draft
**Priority:** P2
**Completed Stories:** 2/3
**Dependencies:** Epic 10 (Centralized Corpus), Epic 6 (Vector Embeddings), Story 6.6 (MCP Corpus Search Server)
**Target Completion:** TBD

---

## Overview

Transform corpus filter agents from passive document filters into interactive personas callable via slash commands. Each agent activates with a role-appropriate personality and scopes its responses to the agent's filtered corpus.

**Problem Statement:**
- Corpus agents currently only filter documents for embeddings
- No interactive persona for specialized assistance
- Users must manually scope queries to relevant collections
- No personality differentiation between agent types

**Solution:**
- Extend agent YAML schema with `persona` block
- Generate `.claude/commands/` files for each corpus agent
- Activate agents via `/ceo`, `/venture-attorney`, etc.
- Automatically scope responses to agent's corpus collections
- Distinct personalities matching professional roles

---

## Goals

1. Enable slash command activation for all corpus agents
2. Define role-appropriate personas for each agent
3. Automatically scope queries to agent's corpus collections
4. Integrate with existing RAG/embeddings infrastructure
5. Maintain clear distinction from BMAD workflow agents
6. Provide generator script for command file maintenance

---

## Stories Breakdown

### Story 11.1: Corpus Agent Slash Commands
- Extend agent YAML schema with persona block
- Create command generator script
- Generate slash commands for all corpus agents
- Define personas for each agent role
- Create `/corpus-agents` listing command
- Update `/agents` to show both agent types

**AC:**
- All corpus agents callable via `/agent-name`
- Personas display appropriate communication style
- Corpus scoping active during agent session
- Generator script produces idempotent output
- Existing BMAD agents unaffected

**Status:** Done ✓

### Story 11.2: Shared Context Architecture
- Create Syncthing-shared folder structure for multi-machine context access
- Migrate agent context documents to shared location
- Establish symlink from project to shared folder
- Document setup process for new machine onboarding

**AC:**
- Shared folder created at `~/Documents/Shared/` with Context and Planning subdirs
- Agent context migrated and accessible via symlink
- Git working tree remains clean (symlink ignored)
- CLAUDE.md updated with architecture documentation
- Setup instructions documented for new machines

**Status:** Draft

### Story 11.3: Multi-Agent Party Mode
- Create `/party` slash command for collaborative multi-agent discussion
- Automatic agent selection (2-4 per message) based on topic-expertise matching
- Each agent searches its own corpus via MCP before responding in-character
- Cross-talk protocol: agents reference, build on, and challenge each other
- Compact roster format to manage token budget across 23 agents
- Graceful degradation for missing embeddings

**AC:**
- `/party` activates multi-agent discussion with welcome roster
- 2-4 agents auto-selected per message with corpus-grounded citations
- Cross-talk between agents with `[corpus]`, `[web]`, or `[training]` citations
- User can request specific agents by name
- Clean exit protocol (no `/exit` collision)
- No new Python code; pure prompt engineering

**Status:** Done ✓

---

## Technical Architecture

### Extended Agent YAML Schema

```yaml
name: ceo
description: Executive leadership agent

persona:
  icon: "🎯"
  style: |
    Communication style description...
  expertise:
    - Area 1
    - Area 2
  greeting: |
    Activation message...

corpus_filter:
  collections:
    - strategy
    - finance
```

### Generated Command Structure

```
.claude/commands/<agent>.md
├── Front matter (description)
├── Persona section (icon, style)
├── Expertise list
├── Corpus scope directive
├── Embeddings path reference
└── Greeting message
```

---

## Dependencies

### Epic Dependencies
- **Epic 10**: Agent filter module and corpus collections
- **Epic 6**: Vector embeddings for RAG integration

### External Dependencies
- None (uses existing Claude Code slash command system)

---

## Implementation Order

```
Story 11.1 (Complete Feature) ✓ DONE
    ├── Schema extension
    ├── Generator script
    ├── Persona definitions
    └── Command file generation

Story 11.2 (Shared Context Architecture)
    ├── Shared folder structure (user configures Syncthing)
    ├── Context document migration
    ├── Symlink creation
    └── Documentation updates
```

Story 11.2 enables multi-machine collaboration for agent context documents via Syncthing.

Story 11.3 (Multi-Agent Party Mode)
    ├── Compact agent roster design (23 agents, ~2K tokens)
    ├── Agent selection logic (topic-expertise matching)
    ├── Per-agent corpus search routing via MCP
    ├── Cross-talk protocol and response formatting
    ├── Session lifecycle (welcome, discussion loop, exit)
    └── /party slash command file

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Naming collision with BMAD agents | Low | Different naming conventions; corpus agents use role names |
| Persona tone inconsistent | Medium | Review all personas for consistency before release |
| Corpus scoping not enforced | Medium | Clear documentation; LLM instruction quality |
| Generator script breaks existing commands | Low | Only generates to `/agents/` namespace; doesn't touch BMAD |
| Party mode token bloat | Medium | Compact roster (~2K tokens); limit `top_k` to 3; cap at 4 agents per round |
| Exit command collision | Low | Use `[E]` notation and phrase triggers; never bare "exit" |
| Agent selection misses relevant experts | Medium | Allow user-directed agent selection as override |

---

## Testing Strategy

### Unit Tests
- YAML schema parsing with persona block
- Command generator output validation
- Idempotency verification

### Integration Tests
- Slash command activation
- Persona greeting display
- Corpus scope instruction presence

### Manual Testing
- Verify each agent activates correctly
- Check personality consistency
- Validate corpus scope behavior

---

## Acceptance Criteria (Epic Level)

1. All 23 corpus agents callable via slash commands
2. Each agent has distinct, role-appropriate persona
3. Corpus scope clearly communicated in agent activation
4. `/corpus-agents` lists available corpus agents
5. `/agents` distinguishes BMAD from corpus agents
6. Generator script documented and functional
7. No regressions to BMAD agent system
8. `/party` enables multi-agent collaborative discussion with corpus-grounded citations
9. Party mode supports automatic and user-directed agent selection

---

## Definition of Done

- Story 11.1 completed and tested ✓
- All agent YAMLs updated with persona blocks ✓
- Generator script functional and documented ✓
- Command files generated for all agents ✓
- Documentation updated (CLAUDE.md, agents.md) ✓
- Manual verification of each agent activation ✓
- Story 11.2 completed: shared context architecture operational
- Symlink from context/ to Syncthing-shared folder functional
- New machine setup documented
- Story 11.3 completed: `/party` slash command functional
- Multi-agent discussion with corpus-grounded citations verified
- Token consumption within budget (< 50K for 5-turn session)
- Exit protocol does not collide with Claude Code commands

---

## Notes

- Corpus agents focus on knowledge retrieval and domain expertise
- BMAD agents focus on workflow and process execution
- The two agent types serve complementary purposes
- Future stories could add: conversation memory, agent handoff, persistent session summaries

---

## References

- [Story 11.1](../stories/11.1-corpus-agent-slash-commands-v02.md) - Corpus agent slash commands (Done)
- [Story 11.2](../stories/11.2-shared-context-architecture-v02.md) - Shared context architecture (Draft)
- [Story 11.3](../stories/11.3-multi-agent-party-mode-v02.md) - Multi-agent party mode (Draft)
- [Epic 10](./epic-10-centralized-corpus-v02.md) - Centralized corpus architecture
- [Epic 6](./epic-6-vector-embeddings-v02.md) - Vector embeddings infrastructure
- [Story 6.6](../stories/6.6-mcp-corpus-search-server-v02.md) - MCP corpus search server
