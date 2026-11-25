# Configuration Generation

`schema.json` defines GlobalConfig, RasterConfig and RaytracingConfig fields,
defaults, TOML paths and renderer enum names. Generated files are checked in for
review. Edit the schema and run:

```sh
python tools/config/generate.py
python tools/config/generate.py --check
python tools/config/test_generate.py
```

CMake requires Python 3.8 or newer. It generates missing files at configure time
and tracks schema/generator changes at build time. Content that has not changed
is not rewritten. Default values live in `.inl` files included by the relevant
`.cpp`; changing a default does not modify public declarations. Adding/removing
fields or changing their types changes declarations and requires rebuilding
their consumers.

All startup settings use `MoerEngine.toml` (or the `--config` override):

```toml
[engine.render.raster]
low_quality_mode = true
aa_mode = "SMAA_1X"
bloom_enabled = false

[engine.render.raster.tonemapping_ae]
enabled = false

[engine.render.raytracing.grid_config]
light_per_ceil = 256
cell_size = 2.0
```

The published `light_per_ceil` name remains the only storage; its correctly
spelled C++ accessors remain available. `grid_config.cell_size` is an initial
value: the existing raytracing renderer fits it to scene bounds when the scene
becomes ready. Renderer enums use case-sensitive names,
vectors use three-element arrays, fixed scalar arrays require their full length,
and probe volumes use up to four `[[engine.render.raster.probe_gi_volumes]]`
tables. Missing fields retain defaults. Wrong types, unknown enum names,
non-finite renderer floats and malformed vectors/scalar arrays retain the prior
default or preset. A parse failure leaves both output configurations unchanged.
Numeric domain validation (for example, a useful exposure or sample-count range)
continues to belong to the renderer. This does not add a live reload mechanism.

Startup order is defaults, the legacy low-quality preset, explicit renderer
fields, then the `enable_shadow=false` master switch. UI construction does not
overwrite loaded settings. The generated template comments renderer defaults
as optional overrides so selecting the quality preset still works. Core fallback
values and the few different template suggestions are intentionally preserved.
Legacy GlobalConfig signed-to-unsigned conversions are also preserved.

Core owns GlobalConfig and has no Render dependency. Render owns its typed
configuration and loads its section from the same file during Engine startup,
before console values bind to EditorConfig. Parsing stays in implementation files;
the renderer configuration headers do not expose TOML parser types. Core and Render
parse their respective configuration separately at startup. Transient UI indices
and one-shot export/framing commands are deliberately not serialized.

Default constructors are out of line, so these configuration structures are no
longer aggregates. Ordinary default construction, field access, copy/move and
the existing nested type names remain available. No repository call sites relied
on positional construction outside the migrated default definitions.

With `moer_build_test=ON`, build `TestConfigGeneration` and run
`ctest --test-dir build -R 'Config(Generation|Codegen)Contract' --output-on-failure`.
