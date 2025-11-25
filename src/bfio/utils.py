# -*- coding: utf-8 -*-
# import core packages
import scyjava
import copy
import logging
from typing import Optional

# Third party packages
import re

from xml.etree import ElementTree as ET
from xsdata.utils.dates import DateTimeParser

from ome_types.model import UnitsLength

KNOWN_INVALID_OME_XSD_REFERENCES = [
    "www.openmicroscopy.org/Schemas/ome/2013-06",
    "www.openmicroscopy.org/Schemas/OME/2012-03",
    "www.openmicroscopy.org/Schemas/sa/2013-06s",
]
REPLACEMENT_OME_XSD_REFERENCE = "www.openmicroscopy.org/Schemas/OME/2016-06"

ome_formats = {
    "detector_id": lambda x: f"Detector:{x}",
    "instrument_id": lambda x: f"Instrument:{x}",
    "image_id": lambda x: f"Image:{x}",
}

try:

    def start() -> str:
        """Start the jvm.

        This function starts the jvm and imports all the necessary Java classes
        to read images using the Bio-Formats toolbox.

        Return:
            The Bio-Formats JAR version.
        """

        global JAR_VERSION
        scyjava.config.endpoints.append("ome:formats-gpl:8.0.1")
        scyjava.start_jvm()
        import loci

        loci.common.DebugTools.setRootLevel("ERROR")
        JAR_VERSION = loci.formats.FormatTools.VERSION

        logging.getLogger("bfio.start").info(
            "bioformats_package.jar version = {}".format(JAR_VERSION)
        )

        return JAR_VERSION

except ModuleNotFoundError:

    def start():  # NOQA: D103
        raise ModuleNotFoundError("Error importing jpype or a loci_tools.jar class.")


def clean_ome_xml_for_known_issues(xml: str) -> str:
    """Clean an OME XML string.

    This was modified from from AICSImageIO:
    https://github.com/AllenCellModeling/aicsimageio/blob/240c1c76a7e884aa37e11a1fbe0fcbb89fea6515/aicsimageio/metadata/utils.py#L187
    """
    # Store list of changes to print out with warning
    metadata_changes = []

    # Fix xsd reference
    # This is from OMEXML object just having invalid reference
    for known_invalid_ref in KNOWN_INVALID_OME_XSD_REFERENCES:
        if known_invalid_ref in xml:
            xml = xml.replace(
                known_invalid_ref,
                REPLACEMENT_OME_XSD_REFERENCE,
            )
            metadata_changes.append(
                f"Replaced '{known_invalid_ref}' with "
                f"'{REPLACEMENT_OME_XSD_REFERENCE}'."
            )

    # Read in XML
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        # remove special char if initial parsing fails
        xml = xml.replace("&#0;", "")
        root = ET.fromstring(xml)

    # Get the namespace
    # In XML etree this looks like
    # "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
    # and must prepend any etree finds
    namespace_matches = re.match(r"\{.*\}", root.tag)
    if namespace_matches is not None:
        namespace = namespace_matches.group(0)
    else:
        raise ValueError("XML does not contain a namespace")

    # Fix MicroManager Instrument and Detector
    instrument = root.find(f"{namespace}Instrument")
    if instrument is not None:
        instrument_id = instrument.get("ID")
        if instrument_id == "Microscope":
            ome_instrument_id = ome_formats["instrument_id"](0)
            instrument.set("ID", ome_instrument_id)
            metadata_changes.append(
                f"Updated attribute 'ID' from '{instrument_id}' to "
                f"'{ome_instrument_id}' on Instrument element."
            )

            for detector_index, detector in enumerate(
                instrument.findall(f"{namespace}Detector")
            ):
                detector_id = detector.get("ID")
                if detector_id is not None:
                    # Create ome detector id if needed
                    ome_detector_id = None
                    if detector_id == "Camera":
                        ome_detector_id = ome_formats["detector_id"](detector_index)
                    elif not detector_id.startswith("Detector:"):
                        ome_detector_id = ome_formats["detector_id"](detector_id)

                    # Apply ome detector id if replaced
                    if ome_detector_id is not None:
                        detector.set("ID", ome_detector_id)
                        metadata_changes.append(
                            f"Updated attribute 'ID' from '{detector_id}' to "
                            f"'{ome_detector_id}' on Detector element at "
                            f"position {detector_index}."
                        )

    # Find all Image elements and fix IDs and refs to fixed instruments
    # This is for certain for test files of o.urs and ACTK files
    for image_index, image in enumerate(root.findall(f"{namespace}Image")):
        image_id = image.get("ID")
        if image_id is not None:
            found_image_id = image_id

            if not found_image_id.startswith("Image"):
                ome_image_id = ome_formats["image_id"](found_image_id)
                image.set("ID", ome_image_id)
                metadata_changes.append(
                    f"Updated attribute 'ID' from '{image_id}' to '{ome_image_id}' "
                    f"on Image element at position {image_index}."
                )
        # Fix bad acquisition date refs
        acq_date_ref = image.find(f"{namespace}AcquisitionDate")
        if acq_date_ref is not None:
            try:
                parser = DateTimeParser(acq_date_ref.text, "%Y-%m-%dT%H:%M:%S%z")
                next(parser.parse())
            except ValueError:
                image.remove(acq_date_ref)
                metadata_changes.append("Removed badly formatted AcquisitionDate.")

        # Fix MicroManager bad instrument refs
        instrument_ref = image.find(f"{namespace}InstrumentRef")
        if instrument_ref is not None:
            instrument_ref_id = instrument_ref.get("ID")
            if instrument_ref_id == "Microscope":
                instrument_ref.set("ID", ome_instrument_id)

        # Find all Pixels elements and fix IDs
        for pixels_index, pixels in enumerate(image.findall(f"{namespace}Pixels")):
            pixels_id = pixels.get("ID")
            if pixels_id is not None:
                found_pixels_id = pixels_id

                if not found_pixels_id.startswith("Pixels"):
                    pixels.set("ID", f"Pixels:{found_pixels_id}")
                    metadata_changes.append(
                        f"Updated attribute 'ID' from '{found_pixels_id}' to "
                        f"Pixels:{found_pixels_id}' on Pixels element at "
                        f"position {pixels_index}."
                    )

            # Determine if there is an out-of-order channel / plane elem
            # This is due to OMEXML "add channel" function
            # That added Channels and appropriate Planes to the XML
            # But, placed them in:
            # Channel
            # Plane
            # Plane
            # ...
            # Channel
            # Plane
            # Plane
            #
            # Instead of grouped together:
            # Channel
            # Channel
            # ...
            # Plane
            # Plane
            # ...
            #
            # This effects all CFE files (new and old) but for different reasons
            pixels_children_out_of_order = False
            encountered_something_besides_channel = False
            encountered_plane = False
            for child in pixels:
                if child.tag != f"{namespace}Channel":
                    encountered_something_besides_channel = True
                if child.tag == f"{namespace}Plane":
                    encountered_plane = True
                if (
                    encountered_something_besides_channel
                    and child.tag == f"{namespace}Channel"
                ):
                    pixels_children_out_of_order = True
                    break
                if encountered_plane and child.tag in [
                    f"{namespace}{t}" for t in ["BinData", "TiffData", "MetadataOnly"]
                ]:
                    pixels_children_out_of_order = True
                    break

            # Ensure order of:
            # channels -> bindata | tiffdata | metadataonly -> planes
            if pixels_children_out_of_order:
                # Get all relevant elems
                channels = [
                    copy.deepcopy(c) for c in pixels.findall(f"{namespace}Channel")
                ]
                bin_data = [
                    copy.deepcopy(b) for b in pixels.findall(f"{namespace}BinData")
                ]
                tiff_data = [
                    copy.deepcopy(t) for t in pixels.findall(f"{namespace}TiffData")
                ]
                # There should only be one metadata only element but to standardize
                # list comprehensions later we findall
                metadata_only = [
                    copy.deepcopy(m) for m in pixels.findall(f"{namespace}MetadataOnly")
                ]
                planes = [copy.deepcopy(p) for p in pixels.findall(f"{namespace}Plane")]

                # Old (2018 ish) cell feature explorer files sometimes contain both
                # an empty metadata only element and filled tiffdata elements
                # Since the metadata only elements are empty we can check this and
                # choose the tiff data elements instead
                #
                # First check if there are any metadata only elements
                if len(metadata_only) == 1:
                    # Now check if _one of_ of the other two choices are filled
                    # ^ in Python is XOR
                    if (len(bin_data) > 0) ^ (len(tiff_data) > 0):
                        metadata_children = list(metadata_only[0])
                        # Now check if the metadata only elem has no children
                        if len(metadata_children) == 0:
                            # If so, just "purge" by creating empty list
                            metadata_only = []

                        # If there are children elements
                        # Return XML and let XMLSchema Validation show error
                        else:
                            return xml

                # After cleaning metadata only, validate the normal behaviors of
                # OME schema
                #
                # Validate that there is only one of bindata, tiffdata, or metadata
                if len(bin_data) > 0:
                    if len(tiff_data) == 0 and len(metadata_only) == 0:
                        selected_choice = bin_data
                    else:
                        # Return XML and let XMLSchema Validation show error
                        return xml
                elif len(tiff_data) > 0:
                    if len(bin_data) == 0 and len(metadata_only) == 0:
                        selected_choice = tiff_data
                    else:
                        # Return XML and let XMLSchema Validation show error
                        return xml
                elif len(metadata_only) == 1:
                    if len(bin_data) == 0 and len(tiff_data) == 0:
                        selected_choice = metadata_only
                    else:
                        # Return XML and let XMLSchema Validation show error
                        return xml
                else:
                    # Return XML and let XMLSchema Validation show error
                    return xml

                # Remove all children from element to be replaced
                # with ordered elements
                for elem in list(pixels):
                    pixels.remove(elem)

                # Re-attach elements
                for channel in channels:
                    pixels.append(channel)
                for elem in selected_choice:
                    pixels.append(elem)
                for plane in planes:
                    pixels.append(plane)

                metadata_changes.append(
                    f"Reordered children of Pixels element at "
                    f"position {pixels_index}."
                )

    # This is a result of dumping basically all experiment metadata
    # into "StructuredAnnotation" blocks
    #
    # This affects new (2020) Cell Feature Explorer files
    #
    # Because these are structured annotations we don't want to mess with anyones
    # besides the AICS generated bad structured annotations
    aics_anno_removed_count = 0
    sa = root.find(f"{namespace}StructuredAnnotations")
    if sa is not None:
        for xml_anno in sa.findall(f"{namespace}XMLAnnotation"):
            # At least these are namespaced
            if xml_anno.get("Namespace") == "alleninstitute.org/CZIMetadata":
                # Get ID because some elements have annotation refs
                # in both the base Image element and all plane elements
                aics_anno_id = xml_anno.get("ID")
                for image in root.findall(f"{namespace}Image"):
                    for anno_ref in image.findall(f"{namespace}AnnotationRef"):
                        if anno_ref.get("ID") == aics_anno_id:
                            image.remove(anno_ref)

                    # Clean planes
                    if image is not None:
                        found_image = image

                        pixels_planes: Optional[ET.Element] = found_image.find(
                            f"{namespace}Pixels"
                        )
                        if pixels_planes is not None:
                            for plane in pixels_planes.findall(f"{namespace}Plane"):
                                for anno_ref in plane.findall(
                                    f"{namespace}AnnotationRef"
                                ):
                                    if anno_ref.get("ID") == aics_anno_id:
                                        plane.remove(anno_ref)

                # Remove the whole etree
                sa.remove(xml_anno)
                aics_anno_removed_count += 1

    # Log changes
    if aics_anno_removed_count > 0:
        metadata_changes.append(
            f"Removed {aics_anno_removed_count} AICS generated XMLAnnotations."
        )

    # If there are no annotations in StructuredAnnotations, remove it
    if sa is not None:
        if len(list(sa)) == 0:
            root.remove(sa)

    # If any piece of metadata was changed alert and rewrite
    if len(metadata_changes) > 0:
        # Register namespace
        ET.register_namespace("", f"http://{REPLACEMENT_OME_XSD_REFERENCE}")

        # Write out cleaned XML to string
        xml = ET.tostring(
            root,
            encoding="unicode",
            method="xml",
        )

    return xml


def pixels_per_cm(
    image_dim_px: int, physical_dim: float, unit: UnitsLength = UnitsLength.CENTIMETER
) -> int:
    """
    Calculate the number of pixels per centimeter based on image dimensions
    and physical size.

    Args:
        image_dim_px (int): The image dimension in pixels.
        physical_dim (float): The physical dimension of the image.
        unit (UnitsLength, optional): The unit of the physical dimension.
        Defaults to UnitsLength.CENTIMETER.

    Returns:
        int: The number of pixels per centimeter.

    Raises:
        ValueError: If an unsupported unit is provided.

    Example:
        >>> pixels_per_cm(1000, 5, UnitsLength.MILLIMETER)
        2000
    """

    # Conversion factors to centimeters
    conversion_factors = {
        # SI Units and SI-derived Units
        UnitsLength.YOTTAMETER: 1e24,  # 1 yottameter = 1e24 centimeters
        UnitsLength.ZETTAMETER: 1e21,  # 1 zettameter = 1e21 centimeters
        UnitsLength.EXAMETER: 1e18,  # 1 exameter = 1e18 centimeters
        UnitsLength.PETAMETER: 1e15,  # 1 petameter = 1e15 centimeters
        UnitsLength.TERAMETER: 1e12,  # 1 terameter = 1e12 centimeters
        UnitsLength.GIGAMETER: 1e9,  # 1 gigameter = 1e9 centimeters
        UnitsLength.MEGAMETER: 1e6,  # 1 megameter = 1e6 centimeters
        UnitsLength.KILOMETER: 1e5,  # 1 kilometer = 1e5 centimeters
        UnitsLength.HECTOMETER: 1e4,  # 1 hectometer = 1e4 centimeters
        UnitsLength.DECAMETER: 1e3,  # 1 decameter = 1e3 centimeters
        UnitsLength.METER: 100,  # 1 meter = 100 centimeters
        UnitsLength.DECIMETER: 10,  # 1 decimeter = 10 centimeters
        UnitsLength.CENTIMETER: 1,  # base unit (centimeters)
        UnitsLength.MILLIMETER: 0.1,  # 1 millimeter = 0.1 centimeters
        UnitsLength.MICROMETER: 1e-4,  # 1 micrometer = 1e-4 centimeters
        UnitsLength.NANOMETER: 1e-7,  # 1 nanometer = 1e-7 centimeters
        UnitsLength.PICOMETER: 1e-10,  # 1 picometer = 1e-10 centimeters
        UnitsLength.FEMTOMETER: 1e-13,  # 1 femtometer = 1e-13 centimeters
        UnitsLength.ATTOMETER: 1e-16,  # 1 attometer = 1e-16 centimeters
        UnitsLength.ZEPTOMETER: 1e-19,  # 1 zeptometer = 1e-19 centimeters
        UnitsLength.YOCTOMETER: 1e-22,  # 1 yoctometer = 1e-22 centimeters
        # SI-derived Units
        # 1 ångström = 1e-8 centimeters
        UnitsLength.LATIN_CAPITAL_LETTER_A_WITH_RING_ABOVE: 1e-8,
        # Imperial Units
        UnitsLength.THOU: 2.54e-3,  # 1 thou (mil) = 0.001 inch = 2.54e-3 centimeters
        UnitsLength.LINE: 2.11667,  # 1 line = 1/12 inch = 2.11667 centimeters
        UnitsLength.INCH: 2.54,  # 1 inch = 2.54 centimeters
        UnitsLength.FOOT: 30.48,  # 1 foot = 12 inches = 30.48 centimeters
        UnitsLength.YARD: 91.44,  # 1 yard = 3 feet = 91.44 centimeters
        UnitsLength.MILE: 160934.4,  # 1 mile = 5280 feet = 160934.4 centimeters
    }

    # Ensure the unit is supported
    if unit not in conversion_factors:
        logger = logging.getLogger("bfio.backends")
        logger.warning(
            f"Unsupported unit '{unit}'."
            f"Supported units are: {', '.join(conversion_factors.keys())}"
        )

    # Convert the physical dimensions to centimeters
    physical_dim_cm = physical_dim * conversion_factors.get(unit, 1)

    # Calculate pixels per centimeter for width and height
    pixels_per_cm = image_dim_px / physical_dim_cm

    return int(pixels_per_cm)
