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
  - Fix, commit `44a262c`:
    - the water surface lies below the ground by the height of the banks, from 0.3 m for a brook to 2 m for a wide river;
    - the bed is 1.5 times the mean depth at its middle, at least 0.3 m;
    - banks rise ~27° from the water to the ground.
  - Measured on world-1's parameters: the ground 4-8 m beyond the water's edge now stands 0.5-0.6 m above the water on streams and ~1-3 m on rivers.
  - Check in game: streams and rivers sit in a bed below their banks, deep enough for lane B's water to fill.

- [ ] **Forest everywhere: meadows full of trees.**
  - Seen: woods on every side, with hardly any open land.
  - Cause: in world-1, meadows (22% of the land) grew ~67 trees/ha, with trees in half of their cells. The scrub mantle at forest edges reached ~120 m into every meadow. In subhumid meadows, which are open for want of rain, its formula blew up and planted closed forest (canopy up to 15).
  - Fix, commit `79c0ea6`:
    - the mantle reaches a quarter of the edge width (tens of metres), with its density capped;
    - it grows only in clearings of forest country.
  - Measured on world-1's parameters: meadows ~4 trees/ha (lone trees and a thin mantle), closed forest 30% → 26% of the land, any trees 56% → 48%.
  - Check in game: meadows read as open grassland with lone trees, and woods have a thin scrub edge. The overall forest share is the owner's choice through the Biomes sliders "Open land" and "Forest needs moisture index above".

- [ ] **Big bare clay "lake beds" that are not flat.**
  - Seen: wide red-brown clay plains in dry country, rising and falling, that no flat water mesh could fill.
  - Cause: these were salt flats, the exposed beds of salt lakes that shrink by evaporation, not lakes. The whole basin up to its spill level was marked salt, climbing the slopes. The largest patch in world-1 spanned 119-188 m of height.
  - Fix, commit `d0e3bb5`:
    - a salt flat covers only the basin floor, up to 3 m above the water left;
    - its ground is flattened to just above the flat's level.
  - Measured on world-1's parameters: salt flat area 2.65 → 0.23 km², heights within 0.3 m.
  - Check in game: bare clay plains are gone except flat salt pans beside salt lakes. Real lakes' clay beds lie entirely under their level_m (still.json). Salt pans could get a white salt look from ecology biome 13 (lane B).

- [ ] **Trees on bare rock.**
  - Seen in Tiles 3D: conifer forest standing on rock and gravel tiles on mountainsides.
  - Cause: the 16 m cells' canopy came from smooth soil fields that do not see the dithered, slope-driven rock tiles.
  - Fix, commit `6da5ecf`: each cell's canopy thins by its share of rock tiles, with gravel counting half. In rocky test blocks, cells over 90% rock that had trees went from 108 to 1 of 415.
  - Check in game: no trees on rock faces or scree.

- [ ] **Low, thick, uniform walls near the coast.**
  - Seen in Tiles 3D (preset v2, corner 14820, 14332): a flat floor at ~1 m walled off from 8–9 m land.
  - Cause: the salt marsh coast type clamped all land within ~180 m of the shore down to about 1 m above the tide, and left a wall where the zone ended.
  - Fix, commit `a21bd44`: the marsh flattening applies only on ground already low. It fades with height above the marsh level and toward the zone's edge, and follows the coast type's smooth weight. At that spot: 0% of ground over 45° (was 8%).
  - Check in game: salt marshes are low tidal flats with no walls around them.
- [ ] **Streams in trenches with sheer walls.**
  - Seen in Tiles 3D (preset v2, corner 9772, 20724): a 4 m stream at the bottom of a 5 m vertical cut.
  - Cause: the bank was not allowed below the macro relief, so where the line ran beside the grid's valley its side stayed a wall.
  - Fix, commit `a21bd44`: the bank rises ~27° until it meets the ground, a small V valley. At that spot the steepest slope is now 31°; river spots with a bank over 45° went from 16% to 7%.
  - Check in game: streams sit in soft-sided beds.

- [ ] **River stubs and lens-shaped bulges at junctions.**
  - Seen in the detail window: a short dead-end branch sticking out of a river, and the bed swelling where a tributary joins.
  - Cause: tributaries of a few cells were traced as channels, and at a junction the grid's flow ran side by side with the receiving channel for several cells, which doubled the bed.
  - Fix, commit `CONF`: the side-by-side stretch is cut, so a tributary meets its channel at an angle at one point; tributaries shorter than 150 m are dropped. On the v2 world, short tributaries went from 160 to 12 (those left end at the sea or a lake) and points running inside another channel from 7807 to 783.
  - Check in game: confluences are single Y joins, with no stubs.

## Known, not fixed yet

- **Mud strip in the shallows beside a sand beach.** A salt-marsh coast type next to a sandy beach puts a brown clay sea floor in the shallow water, which looks odd.
- **Lakes and rivers above sea level show as dry beds.** This is lane B's water, not the generator: it draws only the sea for now.

## Confirmed in game

(none yet)
