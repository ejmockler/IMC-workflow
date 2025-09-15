"""
Blob Visualization Components - Small, composable plotting functions
Each function does ONE thing
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

from src.utils.helpers import (
    add_percentage_labels, canonicalize_pair, clean_axis,
    plot_heatmap_with_dendrogram, create_protein_signature,
    build_pair_color_map
)

def _canonical_pair(sig):
    dominant = sig['dominant_proteins'][:2]
    return '+'.join(sorted(dominant))

def _collect_pairs_from_rois(rois):
    pairs = []
    for roi in rois:
        for sig in roi['blob_signatures'].values():
            pairs.append(_canonical_pair(sig))
    return pairs

def plot_spatial_domains(ax, coords, blob_labels, blob_signatures):
    """Plot spatial distribution of domains with consistent pair colors"""
    blob_compositions = []
    
    # Convert coords to numpy array if it's a list (from JSON)
    if isinstance(coords, list):
        coords = np.array(coords)
    if isinstance(blob_labels, list):
        blob_labels = np.array(blob_labels)
    
    # Build a stable color map for all observed pairs
    all_pairs = []
    for sig in blob_signatures.values():
        dominant = sig['dominant_proteins'][:2]
        canonical = '+'.join(sorted(dominant))
        all_pairs.append(canonical)
    pair_color_map = build_pair_color_map(all_pairs)
    
    valid_blobs = list(blob_signatures.keys())
    for blob_key in valid_blobs:
        sig = blob_signatures[blob_key]
        blob_coords = sig.get('coords')
        if blob_coords is None or len(blob_coords) == 0:
            continue
        # Convert blob_coords to numpy array if needed
        if isinstance(blob_coords, list):
            blob_coords = np.array(blob_coords)
        dominant = sig['dominant_proteins'][:2]
        label_pair = '+'.join(dominant)
        canonical = '+'.join(sorted(dominant))
        color = pair_color_map.get(canonical, 'gray')
        ax.scatter(blob_coords[:, 0], blob_coords[:, 1], c=[color],
                   alpha=0.6, s=1, label=label_pair)
        blob_compositions.append(sig['mean_expression'])
    
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Y position (μm)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=10, fontsize=8)
    ax.set_aspect('equal')
    ax.set_title('Spatial Domain Map')
    
    return blob_compositions

def plot_domain_signatures_heatmap(ax, blob_signatures, protein_names):
    """Plot protein expression signatures as heatmap with proper x-axis labels"""
    blob_compositions = []
    blob_names = []
    
    valid_blobs = sorted(blob_signatures.keys(), 
                        key=lambda x: blob_signatures[x]['size'], 
                        reverse=True)[:15]  # Limit for readability
    
    for blob_key in valid_blobs:
        sig = blob_signatures[blob_key]
        dominant = sig['dominant_proteins'][:2]
        blob_name = '+'.join(dominant)
        blob_names.append(blob_name)
        blob_compositions.append(sig['mean_expression'])
    
    if blob_compositions:
        composition_matrix = np.array(blob_compositions)
        
        # Normalize expression values to 0-1 range to prevent >1 values
        if composition_matrix.max() > 0:
            composition_matrix = composition_matrix / composition_matrix.max()
        
        # Ensure protein names match data dimensions
        actual_n_proteins = composition_matrix.shape[1]
        if len(protein_names) != actual_n_proteins:
            # Truncate or pad protein names to match data
            if len(protein_names) > actual_n_proteins:
                protein_names = protein_names[:actual_n_proteins]
            else:
                # Pad with generic names
                protein_names = protein_names + [f'P{i}' for i in range(len(protein_names), actual_n_proteins)]
        
        # Create heatmap with domain names as x-axis and proteins as y-axis
        im = ax.imshow(composition_matrix.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Set labels
        ax.set_xticks(range(len(blob_names)))
        ax.set_xticklabels(blob_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(protein_names)))
        ax.set_yticklabels(protein_names, fontsize=8)
        ax.set_title('Domain Protein Signatures')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add text annotations for high expression values
        for i in range(len(protein_names)):
            for j in range(len(blob_names)):
                value = composition_matrix[j, i]  # Note: transposed indexing
                if value > 0.6:  # Only show high normalized expression
                    ax.text(j, i, f'{value:.2f}', 
                           ha='center', va='center', fontsize=6, 
                           color='white' if value > 0.8 else 'black',
                           fontweight='bold')

def plot_spatial_contact_matrix(ax, blob_labels, cluster_signatures, coords, radius: float = 15.0, sample_step: int = 3, max_neighbors: int = None, max_samples: int = 20000, blob_type_mapping=None, protein_names=None, return_data: bool = False):
    """Plot adjacency frequency matrix with single proteins on y-axis and colocalized domains on x-axis.
    radius: neighborhood radius in same units as coords
    sample_step: stride when sampling pixels for efficiency
    max_neighbors: optional cap on neighbors per sampled pixel
    blob_type_mapping: dict mapping numeric cluster IDs to domain type strings
    protein_names: list of individual protein names for y-axis
    """
    # Map each blob id -> canonical domain type string (full names, no truncation)
    blob_to_type = {}
    if blob_type_mapping is not None:
        # Use the mapping from numeric cluster IDs to domain types
        for cid, domain_type in blob_type_mapping.items():
            blob_to_type[cid] = domain_type
    else:
        # Fallback: assume cluster_signatures keys match blob_labels
        for cid, sig in cluster_signatures.items():
            dominant = sig['dominant_proteins'][:2]
            blob_to_type[cid] = '+'.join(sorted(dominant))
    
    unique_types = sorted(set(blob_to_type.values()))
    if not unique_types:
        ax.text(0.5, 0.5, 'No domains', ha='center', va='center')
        ax.axis('off')
        return
    
    # Separate colocalized domains (for x-axis) from potential single proteins (for y-axis)
    colocalized_domains = [t for t in unique_types if '+' in t]
    
    # Get single proteins from protein_names or extract from domains
    if protein_names:
        single_proteins = sorted([p.split('(')[0] if '(' in p else p for p in protein_names])
    else:
        # Extract unique proteins from domain names
        all_proteins = set()
        for domain in unique_types:
            all_proteins.update(domain.split('+'))
        single_proteins = sorted(all_proteins)
    
    if not colocalized_domains or not single_proteins:
        ax.text(0.5, 0.5, 'Insufficient data for matrix', ha='center', va='center')
        ax.axis('off')
        return
    
    # Convert coords to numpy array if it's a list (from JSON)
    if isinstance(coords, list):
        coords = np.array(coords)
    if isinstance(blob_labels, list):
        blob_labels = np.array(blob_labels)
    
    # KDTree over pixels
    tree = cKDTree(coords)
    
    # Track contacts: protein -> colocalized_domain -> count
    protein_domain_contacts = defaultdict(lambda: defaultdict(int))
    protein_totals = defaultdict(int)
    
    # Adaptive sampling: cap total samples to max_samples
    step = max(1, sample_step)
    if len(coords) // step > max_samples:
        step = max(1, len(coords) // max_samples)
    sample_indices = np.arange(0, len(coords), step)
    sampled_coords = coords[sample_indices]
    sampled_labels = blob_labels[sample_indices]

    # Bulk neighbor query
    all_neighbors = tree.query_ball_point(sampled_coords, r=radius)
    for i, neighbors in enumerate(all_neighbors):
        cid = sampled_labels[i]
        # Handle both string and integer keys from JSON serialization
        domain = blob_to_type.get(str(cid) if isinstance(cid, (int, np.integer)) else cid)
        if domain is None:
            continue
            
        # Extract proteins from this domain
        proteins_in_domain = domain.split('+')
        
        if max_neighbors is not None:
            neighbors = neighbors[:max_neighbors]
            
        for nb_idx in neighbors:
            if nb_idx == sample_indices[i]:
                continue
            nb_cid = blob_labels[nb_idx]
            # Handle both string and integer keys from JSON serialization
            nb_domain = blob_to_type.get(str(nb_cid) if isinstance(nb_cid, (int, np.integer)) else nb_cid)
            if nb_domain is None or nb_domain == domain:
                continue
            
            # For each protein in current domain, count contact with neighbor domain
            # Only count if neighbor is a colocalized domain
            if '+' in nb_domain:
                for protein in proteins_in_domain:
                    protein_domain_contacts[protein][nb_domain] += 1
                    protein_totals[protein] += 1
    
    # Build asymmetric matrix: proteins (y-axis) x colocalized_domains (x-axis)
    n_proteins = len(single_proteins)
    n_domains = len(colocalized_domains)
    contact_matrix = np.zeros((n_proteins, n_domains))
    
    protein_idx_map = {p: i for i, p in enumerate(single_proteins)}
    domain_idx_map = {d: i for i, d in enumerate(colocalized_domains)}
    
    for protein, domain_contacts in protein_domain_contacts.items():
        if protein in protein_idx_map:
            i = protein_idx_map[protein]
            total = protein_totals.get(protein, 0)
            if total > 0:
                for domain, count in domain_contacts.items():
                    if domain in domain_idx_map:
                        j = domain_idx_map[domain]
                        contact_matrix[i, j] = count / total
    
    # If completely empty, try a single-pass fallback with expanded radius
    if np.all(contact_matrix == 0) and len(sample_indices) > 0:
        expanded_radius = radius * 2
        all_neighbors = tree.query_ball_point(sampled_coords, r=expanded_radius)
        for i, neighbors in enumerate(all_neighbors):
            cid = sampled_labels[i]
            # Handle both string and integer keys from JSON serialization
            domain = blob_to_type.get(str(cid) if isinstance(cid, (int, np.integer)) else cid)
            if domain is None:
                continue
            proteins_in_domain = domain.split('+')
            if max_neighbors is not None:
                neighbors = neighbors[:max_neighbors]
            for nb_idx in neighbors:
                if nb_idx == sample_indices[i]:
                    continue
                nb_cid = blob_labels[nb_idx]
                # Handle both string and integer keys from JSON serialization
                nb_domain = blob_to_type.get(str(nb_cid) if isinstance(nb_cid, (int, np.integer)) else nb_cid)
                if nb_domain is None or nb_domain == domain:
                    continue
                if '+' in nb_domain:
                    for protein in proteins_in_domain:
                        protein_domain_contacts[protein][nb_domain] += 1
                        protein_totals[protein] += 1
        # rebuild matrix
        contact_matrix[:] = 0
        for protein, domain_contacts in protein_domain_contacts.items():
            if protein in protein_idx_map:
                i = protein_idx_map[protein]
                total = protein_totals.get(protein, 0)
                if total > 0:
                    for domain, count in domain_contacts.items():
                        if domain in domain_idx_map:
                            j = domain_idx_map[domain]
                            contact_matrix[i, j] = count / total
    
    # Plot the asymmetric matrix
    if return_data:
        return contact_matrix, single_proteins, colocalized_domains
    
    im = ax.imshow(contact_matrix, cmap='Blues', vmin=0, vmax=max(0.5, contact_matrix.max() or 0.5), aspect='auto')
    ax.set_xticks(range(n_domains))
    ax.set_yticks(range(n_proteins))
    ax.set_xticklabels(colocalized_domains, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(single_proteins, fontsize=8)
    ax.set_xlabel('Colocalized Domains', fontsize=9)
    ax.set_ylabel('Single Proteins', fontsize=9)
    ax.set_title('Protein-Domain Spatial Contacts (Adjacency frequency)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_domain_size_distribution(ax, cluster_signatures, protein_names):
    """Plot domain sizes by marker pair with consistent pair colors"""
    from itertools import combinations
    
    # Build size per pair
    size_per_pair = defaultdict(int)
    observed_pairs = []
    for cid, sig in cluster_signatures.items():
        canonical = '+'.join(sorted(sig['dominant_proteins'][:2]))
        size_per_pair[canonical] += sig['size']
        observed_pairs.append(canonical)
    
    # Get all pairs and filter significant ones
    all_pairs = ['+'.join(sorted(p)) for p in combinations(protein_names, 2)]
    sizes = [size_per_pair.get(p, 0) for p in all_pairs]
    
    # Filter to show only significant pairs
    total_pixels = sum(sizes)
    filtered_data = [(p, s) for p, s in zip(all_pairs, sizes) 
                     if s > 0 and round(100 * s / total_pixels) > 0]
    
    if filtered_data:
        filtered_pairs, filtered_sizes = zip(*filtered_data)
        # Build colors for filtered pairs
        pair_color_map = build_pair_color_map(observed_pairs)
        colors = [pair_color_map.get(p, 'gray') for p in filtered_pairs]
        bars = ax.bar(range(len(filtered_pairs)), filtered_sizes, color=colors)
        ax.set_xticks(range(len(filtered_pairs)))
        ax.set_xticklabels(filtered_pairs, rotation=90, ha='center', fontsize=9)
        ax.set_ylabel('Pixel Count (log scale)')
        ax.set_yscale('log')
        ax.set_title('Domain Sizes by Marker Pair')
        ax.margins(x=0.01, y=0.05)
        
        add_percentage_labels(ax, bars, filtered_sizes, total_pixels)

def plot_top_domain_contacts(ax, blob_contacts, blob_signatures, max_pairs=20):
    """Plot top domain-domain contact pairs with consistent pair colors"""
    # Canonicalize and accumulate contacts
    contact_pair_dict = {}
    unique_regions = set()
    observed_pairs = []
    
    for blob_key, contacts in blob_contacts.items():
        blob1_proteins = '+'.join(blob_signatures[blob_key]['dominant_proteins'][:2])
        unique_regions.add(blob1_proteins)
        for neighbor_key, freq in contacts.items():
            if freq > 0.0:
                blob2_proteins = '+'.join(blob_signatures[neighbor_key]['dominant_proteins'][:2])
                unique_regions.add(blob2_proteins)
                
                # Canonicalize pair
                pair = canonicalize_pair(blob1_proteins, blob2_proteins, ' ↔ ')
                contact_pair_dict[pair] = max(contact_pair_dict.get(pair, 0), freq)
                # Track observed canonical pairs for color mapping
                observed_pairs.append('+'.join(sorted(blob1_proteins.split('+'))))
                observed_pairs.append('+'.join(sorted(blob2_proteins.split('+'))))
    
    if not contact_pair_dict:
        return
    
    # Get top pairs
    contact_pairs = sorted(contact_pair_dict.items(), key=lambda x: x[1], reverse=True)
    top_pairs = contact_pairs[:max_pairs]
    
    if top_pairs:
        pair_names, freqs = zip(*top_pairs)
        # Build colors per side of each pair using pair color map
        pair_color_map = build_pair_color_map(observed_pairs)
        left_colors = []
        right_colors = []
        for pair in pair_names:
            left, right = pair.split(' ↔ ')
            left_colors.append(pair_color_map.get('+'.join(sorted(left.split('+'))), 'gray'))
            right_colors.append(pair_color_map.get('+'.join(sorted(right.split('+'))), 'gray'))
        # Use a horizontal stacked bar illusion by plotting filled rectangles behind text markers
        # Simpler: color bars by left side, and annotate right color as a swatch
        bars = ax.barh(range(len(pair_names)), freqs, color=left_colors, height=0.8)
        ax.set_yticks(range(len(pair_names)))
        ax.set_yticklabels(pair_names, fontsize=7)
        ax.set_xlabel('Contact Frequency')
        
        domains_str = f"{len(unique_regions)} unique domains"
        if len(unique_regions) <= 8:
            domains_str += f": {', '.join(sorted(unique_regions))}"
        ax.set_title(f'Top {len(pair_names)} Domain Contacts\n{domains_str}')
        
        # Add frequency labels
        for i, (bar, freq) in enumerate(zip(bars, freqs)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{freq:.2f}', va='center', fontsize=8)
            # Add right-side color swatch
            ax.plot([0], [0], color=right_colors[i], marker='s', linestyle='None',
                    markersize=6, label='_nolegend_')

def plot_aggregated_contact_matrix(ax, rois):
    """Aggregate adjacency frequencies across ROIs and plot clustered matrix.
    The matrix is over canonical marker pairs (columns/rows aligned).
    """
    from imc_utils import create_contact_matrix, plot_heatmap_with_dendrogram
    # Build unified label set
    labels = sorted(set(_collect_pairs_from_rois(rois)))
    if not labels:
        ax.text(0.5, 0.5, 'No domains found', ha='center', va='center')
        ax.axis('off')
        return
    # Accumulate canonicalized contacts as max frequency across ROIs
    from collections import defaultdict
    accum = defaultdict(float)
    for roi in rois:
        # Build contacts at pair level for this ROI
        for blob_key, contacts in roi['blob_contacts'].items():
            pair1 = '+'.join(sorted(roi['blob_signatures'][blob_key]['dominant_proteins'][:2]))
            for nb_key, freq in contacts.items():
                if freq > 0:
                    pair2 = '+'.join(sorted(roi['blob_signatures'][nb_key]['dominant_proteins'][:2]))
                    key = canonicalize_pair(pair1, pair2, ' ↔ ')
                    accum[key] = max(accum.get(key, 0.0), float(freq))
    # Create and plot
    matrix = create_contact_matrix(accum, labels)
    heat_ax, im = plot_heatmap_with_dendrogram(ax, matrix, labels, title='Aggregated Domain Contacts', cmap='Blues', vmin=0)
    plt.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)
    return heat_ax

def plot_aggregated_domain_signatures(ax, rois, top_n=12):
    """Average domain mean expressions across ROIs for top-N largest domains.
    Reports per canonical pair; averages expression vectors across all matching domains.
    """
    import numpy as np
    from collections import defaultdict
    # Accumulate expressions and sizes by canonical pair
    expr_sums = defaultdict(lambda: None)
    sizes = defaultdict(int)
    protein_names = None
    for roi in rois:
        if protein_names is None:
            protein_names = roi.get('protein_names', protein_names)
        for sig in roi['blob_signatures'].values():
            pair = _canonical_pair(sig)
            sizes[pair] += sig['size']
            vec = np.asarray(sig['mean_expression'])
            expr_sums[pair] = vec if expr_sums[pair] is None else (expr_sums[pair] + vec)
    if not sizes:
        ax.text(0.5, 0.5, 'No domain signatures', ha='center', va='center')
        ax.axis('off')
        return
    # Select top-N by total size
    top_pairs = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:top_n]
    blob_names = [p for p, _ in top_pairs]
    compositions = []
    for p, _ in top_pairs:
        vec = expr_sums[p] / max(1, rois.__len__())
        compositions.append(vec)
    if compositions:
        composition_matrix = np.array(compositions)
        if composition_matrix.max() > 0:
            composition_matrix = composition_matrix / composition_matrix.max()
        if protein_names is None:
            protein_names = [f'P{i}' for i in range(composition_matrix.shape[1])]
        im = ax.imshow(composition_matrix.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(blob_names)))
        ax.set_xticklabels(blob_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(protein_names)))
        ax.set_yticklabels(protein_names, fontsize=8)
        ax.set_title('Aggregated Domain Protein Signatures')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_tissue_neighborhoods(ax, coords, neighborhoods, blob_signatures):
    """Plot tissue neighborhoods with distinct colors and boundaries."""
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    
    if isinstance(coords, list):
        coords = np.array(coords)
    
    if neighborhoods is None:
        ax.text(0.5, 0.5, 'No neighborhoods detected', ha='center', va='center')
        ax.axis('off')
        return
    
    # Handle different neighborhood data structures
    if isinstance(neighborhoods, dict) and 'pixel_assignments' in neighborhoods:
        pixel_assignments = neighborhoods['pixel_assignments']
        n_neighborhoods = neighborhoods.get('n_neighborhoods', 0)
        
        if n_neighborhoods == 0:
            ax.text(0.5, 0.5, 'No neighborhoods detected', ha='center', va='center')
            ax.axis('off')
            return
        
        # Create colormap for neighborhoods
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0', 
                  '#ffb366', '#66ffb3', '#ff6666', '#9999ff']
        
        # Plot each neighborhood
        for neighborhood_id in range(n_neighborhoods):
            color = colors[neighborhood_id % len(colors)]
            
            # Get pixels assigned to this neighborhood
            mask = pixel_assignments == neighborhood_id
            if np.any(mask):
                neighborhood_coords = coords[mask]
                
                # Plot neighborhood pixels
                ax.scatter(neighborhood_coords[:, 0], neighborhood_coords[:, 1], 
                          c=color, alpha=0.6, s=1, label=f'N{neighborhood_id}')
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Tissue Neighborhoods')
        if n_neighborhoods < 10:
            ax.legend(fontsize=8)
        ax.set_aspect('equal')
    
    else:
        ax.text(0.5, 0.5, 'Unsupported neighborhood format', ha='center', va='center')
        ax.axis('off')


def plot_neighborhood_entropy_map(ax, coords, entropy_map):
    """Plot spatial entropy map showing tissue organization boundaries."""
    if isinstance(coords, list):
        coords = np.array(coords)
    
    if entropy_map is None:
        ax.text(0.5, 0.5, 'No entropy map available', ha='center', va='center')
        ax.axis('off')
        return
    
    # Handle different entropy map formats
    if isinstance(entropy_map, dict):
        # Dictionary format: {(x, y): entropy_value}
        x_coords = [pos[0] for pos in entropy_map.keys()]
        y_coords = [pos[1] for pos in entropy_map.keys()]
        entropy_values = list(entropy_map.values())
    elif isinstance(entropy_map, (list, tuple)) and len(entropy_map) >= 3:
        # Tuple format: (x_coords, y_coords, entropy_values)
        x_coords, y_coords, entropy_values = entropy_map[:3]
    else:
        ax.text(0.5, 0.5, 'Unsupported entropy format', ha='center', va='center')
        ax.axis('off')
        return
    
    if not entropy_values or len(entropy_values) == 0:
        ax.text(0.5, 0.5, 'No entropy data', ha='center', va='center')
        ax.axis('off')
        return
    
    # Create scatter plot with entropy as color
    scatter = ax.scatter(x_coords, y_coords, c=entropy_values, 
                        cmap='viridis', alpha=0.7, s=20)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Spatial Entropy Map')
    ax.set_aspect('equal')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label='Shannon Entropy')


def plot_neighborhood_composition(ax, neighborhoods, blob_signatures):
    """Plot neighborhood composition showing dominant protein pairs."""
    if neighborhoods is None:
        ax.text(0.5, 0.5, 'No neighborhoods detected', ha='center', va='center')
        ax.axis('off')
        return
    
    # Handle the new simplified neighborhood format
    if isinstance(neighborhoods, dict) and 'neighborhoods' in neighborhoods:
        neighborhood_data = neighborhoods['neighborhoods']
        n_neighborhoods = neighborhoods.get('n_neighborhoods', 0)
        
        if n_neighborhoods == 0:
            ax.text(0.5, 0.5, 'No neighborhoods detected', ha='center', va='center')
            ax.axis('off')
            return
        
        # Extract dominant protein pairs from each neighborhood
        neighborhood_labels = []
        neighborhood_sizes = []
        dominant_pairs = []
        
        for nid in range(n_neighborhoods):
            if nid in neighborhood_data:
                nbhd = neighborhood_data[nid]
                neighborhood_labels.append(f'N{nid}')
                neighborhood_sizes.append(nbhd['size'])
                
                # Get top dominant pairs
                if nbhd.get('dominant_pairs'):
                    dominant_pairs.append(nbhd['dominant_pairs'][0])  # Top pair
                else:
                    dominant_pairs.append('Mixed')
        
        if not neighborhood_sizes:
            ax.text(0.5, 0.5, 'No composition data', ha='center', va='center')
            ax.axis('off')
            return
        
        # Create bar plot with colors based on dominant pairs
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(dominant_pairs))))
        pair_colors = {pair: colors[i] for i, pair in enumerate(set(dominant_pairs))}
        
        bar_colors = [pair_colors[pair] for pair in dominant_pairs]
        bars = ax.bar(range(len(neighborhood_labels)), neighborhood_sizes, 
                     color=bar_colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, size, pair in zip(bars, neighborhood_sizes, dominant_pairs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(neighborhood_sizes)*0.01,
                   f'{size}\n{pair}', ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Neighborhood')
        ax.set_ylabel('Size (pixels)')
        ax.set_title('Neighborhood Composition')
        ax.set_xticks(range(len(neighborhood_labels)))
        ax.set_xticklabels(neighborhood_labels)
        
        # Add legend for dominant pairs
        unique_pairs = list(set(dominant_pairs))
        if len(unique_pairs) <= 6:  # Only show legend if not too many pairs
            legend_elements = [plt.Rectangle((0,0),1,1, color=pair_colors[pair], alpha=0.8) 
                              for pair in unique_pairs]
            ax.legend(legend_elements, unique_pairs, loc='upper right', fontsize=7)
    
    else:
        ax.text(0.5, 0.5, 'Unsupported neighborhood format', ha='center', va='center')
        ax.axis('off')


def plot_neighborhood_statistics(ax, neighborhoods):
    """Plot neighborhood size distribution and statistics."""
    if neighborhoods is None:
        ax.text(0.5, 0.5, 'No neighborhoods detected', ha='center', va='center')
        ax.axis('off')
        return
    
    # Handle the new simplified neighborhood format
    if isinstance(neighborhoods, dict) and 'neighborhoods' in neighborhoods:
        neighborhood_data = neighborhoods['neighborhoods']
        n_neighborhoods = neighborhoods.get('n_neighborhoods', 0)
        coverage = neighborhoods.get('coverage', 0)
        
        if n_neighborhoods == 0:
            ax.text(0.5, 0.5, 'No neighborhoods detected', ha='center', va='center')
            ax.axis('off')
            return
        
        # Extract neighborhood sizes and blob counts
        sizes = []
        blob_counts = []
        
        for nid in range(n_neighborhoods):
            if nid in neighborhood_data:
                nbhd = neighborhood_data[nid]
                sizes.append(nbhd['size'])
                blob_counts.append(nbhd['blob_count'])
        
        if not sizes:
            ax.text(0.5, 0.5, 'No size data', ha='center', va='center')
            ax.axis('off')
            return
        
        # Create histogram of sizes
        ax.hist(sizes, bins=min(8, max(2, len(sizes))), color='lightsteelblue', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Neighborhood Size (pixels)')
        ax.set_ylabel('Count')
        ax.set_title('Neighborhood Size Distribution')
        
        # Add comprehensive statistics text
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        mean_blobs = np.mean(blob_counts) if blob_counts else 0
        
        stats_text = f'Neighborhoods: {n_neighborhoods}\nMean size: {mean_size:.0f}±{std_size:.0f} px\nMean blobs: {mean_blobs:.1f}\nCoverage: {coverage:.1%}'
        ax.text(0.98, 0.98, stats_text, 
                transform=ax.transAxes, va='top', ha='right', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    else:
        ax.text(0.5, 0.5, 'Unsupported neighborhood format', ha='center', va='center')
        ax.axis('off')

