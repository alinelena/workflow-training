# AiiDA-MLIP Registered Entry Points

Registered entry points exposed by `aiida-mlip` in the AiiDA registry:

## Calculations (`aiida.calculations`)

| Entry Point | Factory Call | Implementation Class |
| :--- | :--- | :--- |
| `mlip.descriptors` | `CalculationFactory('mlip.descriptors')` | `aiida_mlip.calculations.descriptors:Descriptors` |
| `mlip.md` | `CalculationFactory('mlip.md')` | `aiida_mlip.calculations.md:MD` |
| `mlip.opt` | `CalculationFactory('mlip.opt')` | `aiida_mlip.calculations.geomopt:GeomOpt` |
| `mlip.ph` | `CalculationFactory('mlip.ph')` | `aiida_mlip.calculations.phonon:Phonons` |
| `mlip.sp` | `CalculationFactory('mlip.sp')` | `aiida_mlip.calculations.singlepoint:Singlepoint` |
| `mlip.train` | `CalculationFactory('mlip.train')` | `aiida_mlip.calculations.train:Train` |

## Data Types (`aiida.data`)

| Entry Point | Factory Call | Implementation Class |
| :--- | :--- | :--- |
| `mlip.config` | `DataFactory('mlip.config')` | `aiida_mlip.data.config:JanusConfigfile` |
| `mlip.modeldata` | `DataFactory('mlip.modeldata')` | `aiida_mlip.data.model:ModelData` |

## Parsers (`aiida.parsers`)

| Entry Point | Implementation Class |
| :--- | :--- |
| `mlip.descriptors_parser` | `aiida_mlip.parsers.descriptors_parser:DescriptorsParser` |
| `mlip.md_parser` | `aiida_mlip.parsers.md_parser:MDParser` |
| `mlip.opt_parser` | `aiida_mlip.parsers.opt_parser:GeomOptParser` |
| `mlip.ph_parser` | `aiida_mlip.parsers.ph_parser:PhononParser` |
| `mlip.sp_parser` | `aiida_mlip.parsers.sp_parser:SPParser` |
| `mlip.train_parser` | `aiida_mlip.parsers.train_parser:TrainParser` |
