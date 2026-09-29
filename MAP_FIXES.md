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

- [ ] **Rivers not carved: a gravel band level with the hillside.**
  - Seen: river beds painted as gravel but flat with the land around them, so no water mesh could sit in them.
  - Cause: each line's water surface was the ground under it, and the bank rule only lowered land. In world-1, streams up to 8 m wide lay 0.1-0.2 m below the ground beside them, with beds 0.1-0.4 m deep.
  - Fix, commit `4b1c0e3`:
    - the water surface lies below the ground by the height of the banks, from 0.3 m for a brook to 2 m for a wide river;
    - the bed is 1.5 times the mean depth at its middle, at least 0.3 m;
    - banks rise ~27° from the water to the ground.
  - Measured on world-1's parameters: the ground 4-8 m beyond the water's edge now stands 0.5-0.6 m above the water on streams and ~1-3 m on rivers.
  - Check in game: streams and rivers sit in a bed below their banks, deep enough for lane B's water to fill.

## Known, not fixed yet

- **Mud strip in the shallows beside a sand beach.** A salt-marsh coast type next to a sandy beach puts a brown clay sea floor in the shallow water, which looks odd.
- **Lakes and rivers above sea level show as dry beds.** This is lane B's water, not the generator: it draws only the sea for now.

## Confirmed in game

(none yet)
