# Flatland Challenge - Multi-Agent Pathfinding

A comprehensive solution to the Flatland Railway Scheduling Challenge, implementing progressively complex pathfinding algorithms for train routing in railway networks.

## Overview

This project tackles three increasingly complex challenges in multi-agent pathfinding for railway systems:

1. **Single-Agent Pathfinding** - Basic navigation without collision concerns
2. **Multi-Agent Temporal Planning** - Collision-free routing with dynamic obstacles
3. **Full-Scale Scheduling** - Real-world constraints including deadlines, malfunctions, and replanning

## Documentation

For detailed technical analysis, algorithm design decisions, and experimental results, please refer to the **[Flatland Challenge Scientific Report](flatland%20challenge-scientific%20report.pdf)**.

## Challenges

### Question 1: Single-Agent Pathfinding

**Objective:** Find an optimal path from start to goal for a single agent without collision or time constraints.

**Algorithm:** A\* Search with Manhattan Distance Heuristic

**Key Features:**

- Priority queue-based exploration
- Optimal path guarantee
- Direction-aware transitions
- Respects railway grid constraints

### Question 2: Multi-Agent Temporal Planning

**Objective:** Plan paths for multiple agents that avoid vertex and edge conflicts while respecting existing paths.

**Algorithm:** Time-Expanded A\* (SIPP-style)

**Key Features:**

- Temporal conflict avoidance (vertex and edge)
- Reservation table for occupied cells
- Wait actions to avoid collisions
- Head-to-head collision detection
- Sequential planning with priority ordering

### Question 3: Full-Scale Multi-Agent Scheduling

**Objective:** Solve real-world railway scheduling with deadlines, malfunctions, and dynamic replanning.

**Algorithm:** Combined approach using:

- SIPP A\* with unified reservations
- Large Neighborhood Search (LNS) optimization
- Corridor detection and conflict resolution
- Deadlock detection and resolution

**Key Features:**

- **Deadline Management:** Agents prioritized by slack (remaining time vs. distance)
- **Malfunction Handling:** Dynamic replanning when trains experience delays
- **Corridor Detection:** Identifies narrow passages and prevents head-on collisions
- **Unified Reservation System:** Manages vertex, edge, and corridor conflicts
- **Deadlock Resolution:** Detects circular wait patterns and resolves them
- **Memory Optimization:** Efficient reservation tracking with bounded horizons
- **Adaptive Planning:** Scales algorithm parameters based on problem size
