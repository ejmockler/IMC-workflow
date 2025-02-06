flowchart TD

    %% -- Data Input --
    subgraph Input[Data Input]
        direction TB
        subgraph RDataSets[Load R Datasets]
            A1[Load SPE RDS]:::inputNode
            A2[Load Images RDS]:::inputNode
            A3[Load Masks RDS]:::inputNode
        end
        A4[Load Panel CSV]:::inputNode
        A5[Read IMC TXT Files]:::inputNode
    end

    %% -- Data Validation --
    subgraph Valid[Data Validation]
        direction TB
        V1[Initial Validation]:::validationNode
        B1[Validate File Paths]:::validationNode
        B2[Extract IMC Metadata]:::validationNode
        B3[Extract SPE Metadata]:::validationNode
        B4[Validate Markers]:::validationNode
        B5[Map Samples to Files]:::validationNode

        %% Clarify validation steps
        %% Initial checks of file existence and paths
        V1 --> B1
        %% Extract metadata for processing
        subgraph Metadata[Metadata Extraction]
            B2
            B3
        end
        %% Validate consistency and map data appropriately
        Metadata --> B4
        B4 --> B5
    end

    %% -- Visualization Pipeline --
    subgraph Visual[Visualization Pipeline]
        direction TB
        C1[Extract Channel Data]:::visualNode
        C2[Normalize Intensities]:::visualNode
        C3[Plot Channel Images]:::visualNode
        C4[Add Cell Overlays]:::visualNode
        C5[Create Composites]:::visualNode
        
        %% Transform data information to visualization
        C1 --> C2
        C2 --> C3
        %% Enhance visualization with overlays
        C3 --> C4
        %% Final composite creation
        C4 --> C5
    end

    %% -- Analysis Output --
    subgraph Output[Analysis Output]
        direction TB
        D1[Calculate kNN]:::outputNode
        D2[Export PDFs]:::outputNode
        
        %% Perform final analysis and export results
        D1 --> D2
    end

    %% Connections between phases
    RDataSets --> V1
    A4 --> B1
    A5 --> B2

    B5 --> C1

    C5 --> D1

    %% Style Definitions
    classDef inputNode fill:#e0f7fa,stroke:#0097a7,stroke-width:2px;
    classDef validationNode fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px;
    classDef visualNode fill:#FFE0B2,stroke:#FF6F00,stroke-width:2px;
    classDef outputNode fill:#C8E6C9,stroke:#388E3C,stroke-width:2px;
