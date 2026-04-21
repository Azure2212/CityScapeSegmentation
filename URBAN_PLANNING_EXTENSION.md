# Urban Planning Intelligence Extension

This extension is additive. It does not change the original Flask app, routes,
template, model-loading code, training code, or inference behavior. The original
demo remains available from `app.py`; the companion dashboard is launched from
`urban_planning_app.py`.

## Run

```bash
python urban_planning_app.py
```

Then open:

- `http://127.0.0.1:5000/` for the original segmentation demo.
- `http://127.0.0.1:5000/urban` for the urban-planning companion dashboard.

Because this entrypoint imports the existing app module, it preserves the
current startup behavior, including the existing pretrained-weight downloads.

## What The Extension Adds

- A new `/urban` dashboard for planning-oriented analysis.
- A new `/urban-predict` endpoint that returns the segmentation overlay plus
  structured urban-scene metrics.
- A new `utils/urban_scene_analysis.py` module that converts the predicted mask
  into class coverage, planning groups, object counts, heuristic scores, spatial
  context flags, and a concise planning summary.

## Planning Signals

The analysis groups CityScapes classes into urban-planning concepts:

- Mobility surface: road, sidewalk.
- Built environment: building, wall, fence, pole, traffic light, traffic sign.
- Green and open view: vegetation, terrain, sky.
- Vehicles: car, truck, bus, train, motorcycle, bicycle.
- People and active mobility: person, rider, bicycle.

The dashboard also reports object counts for people, riders, cars, trucks,
buses, trains, motorcycles, and bicycles. Counts are derived from connected
components in the predicted mask and ignore tiny blobs below the configured
minimum area.

## Interpretation Limits

The planning scores are transparent heuristics based on semantic segmentation
output. They are useful for demonstrating how pixel-level AI predictions can be
converted into higher-level scene intelligence, but they are not surveyed urban
planning measurements or ground truth.

## Tests

```bash
python -m unittest discover -s tests
```

The tests use synthetic masks to validate class percentages, planning-group
aggregation, connected-component counts, empty-count edge cases, and score
bounds.
