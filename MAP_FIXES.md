# Map fixes to verify

These are fixes made in the generator after a map was imported and seen in the game. Each one needs checking in the game after the next export and import. Tick an item once it is confirmed in the game. Move it back to open, with a note, if it is not.

## Waiting for the next export (after world-1)

- [ ] **Ponds with vertical walls at the shore.**
  - Seen: a dry pond bed with a sheer rock wall of several metres around part of it; some lakes had smooth shores and others did not.
  - Cause: the pond bowl was cut only inside its shoreline. In world-1, 81% of the ponds, every tarn and 95% of the spring pools had a wall steeper than 45°.
  - Fix: a bank rising ~17° from the water until it meets the ground, including the inner side of oxbow crescents. Tarns now need a real hollow. Commit `2912b1c`.
  - Check in game: pond, tarn and oxbow shores slope down into the bed with no step. Until lane B draws still water, the beds still show dry.
- [ ] **Rock walls (cliff bands) too vertical, straight and regular; grassy pillars; grass stripes on rock faces.**
  - Seen: near the coast, near-vertical walls with straight edges following the contours at one spacing, walls ending in sheer grass-topped pillars, and stripes of grass on the rock faces.
  - Fix, commit `f0f2480`:
    - faces at most ~55°, with a rounded top and foot (relief knob "rock wall steepness");
    - lines wander ~60 m across the slope, with re-entrants;
    - layers of uneven thickness;
    - walls die out at their ends as slopes;
    - tors have gentler sides;
    - a tile steeper than ~48° between its own corners is bare rock.
  - Check in game: cliff bands read as sinuous escarpments with steep but not sheer faces, their faces are rock with no grass stripes, and there are no free-standing pillars.

## Known, not fixed yet

- **Mud strip in the shallows beside a sand beach.** A salt-marsh coast type next to a sandy beach puts a brown clay sea floor in the shallow water, which looks odd.
- **Lakes and rivers above sea level show as dry beds.** This is lane B's water, not the generator: it draws only the sea for now.

## Confirmed in game

(none yet)
