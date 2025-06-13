import numpy as np
import pyvista as pv
import xml.etree.ElementTree as ET
import tempfile

# --- Model Definitions ---
MODELS = {
    "gray_scott": {
        "rule_name": "Gray-Scott",
        "formula": (
            "delta_a = D_a * laplacian_a - a*b*b + alpha * (1.0f - a);\n"
            "delta_b = D_b * laplacian_b + a*b*b - (alpha + beta) * b;"
        ),
    },
    "bruss": {
        "rule_name": "Brusselator",
        "formula": (
            "delta_a = D_a * laplacian_a + alpha - (beta + 1) * a + a * b * b;\n"
            "delta_b = D_b * laplacian_b + beta * a - a * b * b;"
        ),
    },
}


def _create_initial_pattern_generator_node() -> ET.Element:
    """Creates the static <initial_pattern_generator> XML node."""
    gen = ET.Element("initial_pattern_generator", zero_first="true")

    # Define initial state: a=1, b=0 everywhere.
    overlay1 = ET.SubElement(gen, "overlay", chemical="a")
    ET.SubElement(overlay1, "overwrite")
    ET.SubElement(overlay1, "constant", value="1")
    ET.SubElement(overlay1, "everywhere")

    overlay2 = ET.SubElement(gen, "overlay", chemical="b")
    ET.SubElement(overlay2, "overwrite")
    ET.SubElement(overlay2, "constant", value="0")
    ET.SubElement(overlay2, "everywhere")

    # Add a patch of noise to 'b' to start the reaction.
    overlay3 = ET.SubElement(gen, "overlay", chemical="b")
    ET.SubElement(overlay3, "overwrite")
    ET.SubElement(overlay3, "constant", value="1.0")
    rect = ET.SubElement(overlay3, "rectangle")
    ET.SubElement(rect, "point3D", x="0.375", y="0.375", z="0.5")
    ET.SubElement(rect, "point3D", x="0.625", y="0.625", z="0")

    return gen


def _create_render_settings_node() -> ET.Element:
    """Creates the static <render_settings> XML node."""
    settings = ET.Element("render_settings")
    ET.SubElement(settings, "active_chemical", value="b")
    # <show_displacement_mapped_surface value="false"/>
    ET.SubElement(settings, "show_displacement_mapped_surface", value="false")
    return settings


def create_vti_input(
    output_filename: str,
    model_name: str,
    Du: float,
    Dv: float,
    alpha: float,
    beta: float,
    domain_dimensions: tuple[int, int, int],
    timestep: float,
    initialization_mode: str = "generator",
    initial_a: np.ndarray = None,
    initial_b: np.ndarray = None,
):
    """
    Generates a VTI input file for a reaction-diffusion simulator.

    Args:
        output_filename: Path for the output .vti file.
        model_name: The name of the reaction-diffusion model (e.g., 'Gray-Scott').
        Du, Dv, alpha, beta: Model-specific floating point parameters.
        domain_dimensions: A tuple (nx, ny, nz) for the simulation grid size.
        timestep: The simulation timestep.
        initialization_mode: 'generator' to use the XML pattern generator,
                             'image_data' to use numpy arrays.
        initial_a, initial_b: NumPy arrays for initial chemical concentrations.
                              Required if mode is 'image_data'.
    """
    # --- 1. Validate Inputs ---
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' is not defined.")

    if initialization_mode not in ["generator", "image_data"]:
        raise ValueError("initialization_mode must be 'generator' or 'image_data'.")

    if initialization_mode == "image_data" and (initial_a is None or initial_b is None):
        raise ValueError(
            "initial_a and initial_b arrays are required for 'image_data' mode."
        )

    # --- 2. Create PyVista Grid and set data if applicable ---
    grid = pv.ImageData()
    grid.dimensions = domain_dimensions
    grid.spacing = (1, 1, 1)  # Voxel size
    grid.origin = (0, 0, 0)

    apply_when_loading = "true"
    if initialization_mode == "image_data":
        apply_when_loading = "false"
    else:
        # Default initial values for generator mode
        if initial_a is None:
            initial_a = np.zeros(domain_dimensions)
        if initial_b is None:
            initial_b = np.zeros(domain_dimensions)

    grid["a"] = initial_a.flatten().astype(np.float32)
    grid["b"] = initial_b.flatten().astype(np.float32)
        # --- 3. Generate base VTK XML from PyVista using a temporary file ---
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=True) as tmp_file:
        # Save the pyvista grid to the temporary file
        grid.save(tmp_file.name, binary=True)

        # Parse the temporary XML file to get the tree root
        vtk_tree = ET.parse(tmp_file.name)
        vtk_tree_root = vtk_tree.getroot()

    # --- 4. Create the custom <RD> block ---
    model = MODELS[model_name]
    rd_node = ET.Element("RD", format_version="6")

    ET.SubElement(rd_node, "description")
    rule = ET.SubElement(
        rd_node,
        "rule",
        name=model["rule_name"],
        type="formula",
        wrap="1",
        neighborhood_type="vertex",
    )

    # Add parameters
    ET.SubElement(rule, "param", name="timestep").text = f"{timestep:.6f}"
    ET.SubElement(rule, "param", name="D_a").text = f"{Du:.6f}"
    ET.SubElement(rule, "param", name="D_b").text = f"{Dv:.6f}"
    ET.SubElement(rule, "param", name="alpha").text = f"{alpha:.6f}"
    ET.SubElement(rule, "param", name="beta").text = f"{beta:.6f}"

    ET.SubElement(
        rule,
        "formula",
        number_of_chemicals="2",
        block_size_x="4",
        block_size_y="1",
        block_size_z="1",
        accuracy="medium",
    ).text = model["formula"]

    # Add generator or use data based on mode
    if initialization_mode == "generator":
        gen_node = _create_initial_pattern_generator_node()
        gen_node.set("apply_when_loading", apply_when_loading)
        rd_node.append(gen_node)

    # Add other static blocks
    rd_node.append(_create_render_settings_node())

    # --- 5. Insert the <RD> block into the PyVista-generated tree ---
    vtk_tree_root.insert(0, rd_node)

    # --- 6. Write the final, combined XML to file ---
    final_tree = ET.ElementTree(vtk_tree_root)
    ET.indent(final_tree, space="  ")
    final_tree.write(output_filename, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    # --- Example 1: Using the internal pattern generator ---
    print("--- Running Example 1: Generator Mode ---")
    create_vti_input(
        output_filename="grayscott_from_generator.vti",
        model_name="Gray-Scott",
        Du=0.14,
        Dv=0.06,
        alpha=0.035,
        beta=0.065,  # Example params for "worms"
        domain_dimensions=(128, 128, 1),
        timestep=1.0,
        initialization_mode="generator",
    )

    # --- Example 2: Using NumPy arrays for initial data ---
    print("\n--- Running Example 2: Image Data Mode ---")
    dims = (128, 128, 1)

    # Create initial state: U=1.0 everywhere, V=0.0 everywhere
    u_initial = np.ones(dims)
    v_initial = np.zeros(dims)

    # Add a small square of V in the middle to kickstart the reaction
    v_initial[48:80, 48:80, :] = 1.0

    create_vti_input(
        output_filename="grayscott_from_imagedata.vti",
        model_name="Gray-Scott",
        Du=0.14,
        Dv=0.06,
        alpha=0.035,
        beta=0.065,  # Example params for "worms"
        domain_dimensions=dims,
        timestep=1.0,
        initialization_mode="image_data",
        initial_a=u_initial,
        initial_b=v_initial,
    )
