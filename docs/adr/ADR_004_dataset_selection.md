# ADR-004 | Dataset Selection

## Status

Accepted 

## Context 

The project requiers a realistic and publicly available dataset to develop, evaluate and demonstrate a predicive maintenance pipeline.

Several options were considered : 

- Simulated electric motor data
- Public wind turbin datasets (ODRE)
- Real Chailift operational data
- NASA C-MAPSS turbofan engine dataset 

## Decision 

Use the NASA C-MAPSS dataset as the primary data source for the project.

## Rationale

The NASA C-MAPSS datset is a well-established benchmark in predictive maintenance research.

It provides : 
- Progressive degradation until failure 
- Multiple operatioal scenarios 
- Remaining Useful Life (RUL) labels
- Extensive scientifique literature
- Public availability
- High reproducibility

## Consequences

### Positive

- Reprocible experiments
- Easy comparison with published reseach 
- Reliable foundation for the project 

### Negative

- Simulated rather than industrial data
- Limited contextual information about the physical sensors 


