import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion


class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/can.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )


class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/milk.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bread.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/milk-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bread-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/can-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in PegInHole)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/plate-with-hole.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class DoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name, friction=None, damping=None, lock=False):
        xml_path = "objects/door.xml"
        if lock:
            xml_path = "objects/door_lock.xml"
        super().__init__(
            xml_path_completion(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.door_body = self.naming_prefix + "door"
        self.frame_body = self.naming_prefix + "frame"
        self.latch_body = self.naming_prefix + "latch"
        self.hinge_joint = self.naming_prefix + "hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic


class CustomTriangleObject(MujocoXMLObject):
    """
    Custom triangle block (used in House)
    """    
    def __init__(self, name, rgba=None):
        """
        Args:
            name (str): Name of the object
            rgba (list/tuple): 4-element (r,g,b,a) color specification
        """
        # Default color if none provided
        default_rgba = [0.5, 0.5, 0.5, 1]
        self.rgba = rgba if rgba is not None else default_rgba
    
        super().__init__(
            xml_path_completion("custom_objects/triangle_block/triangle_block.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True
        )
    
    def _get_object_subtree(self):
        # Get the base XML tree
        tree = super()._get_object_subtree()
        
        # # Find the material and update its rgba
        # print("Available materials:", [elem.attrib for elem in tree.findall(".//material")])
        
        # import xml.etree.ElementTree as ET
        # print(ET.tostring(tree, encoding='unicode'))  # Print the entire XML

        # material = tree.find(".//material[@name='triangle_mat']")
        # if material is not None:
        #     material.set("rgba", " ".join(map(str, self.rgba)))
        
        # Ensure geom uses the updated material
        geom = tree.find(".//geom[@mesh='triangle_mesh']")
        if geom is not None:
            geom.set("material", "triangle_mat")
            # Force color by adding rgba directly to geom as well
            geom.set("rgba", " ".join(map(str, self.rgba)))
        
        # Ensure the visual geom uses the updated material
        geom_visual = tree.find(".//geom[@name='g0_visual']")
        if geom_visual is not None:
            geom_visual.set("material", "triangle_mat")
            # Force color by adding rgba directly to geom as well
            geom_visual.set("rgba", " ".join(map(str, self.rgba)))
        
        return tree

class CustomPentagonObject(MujocoXMLObject):
    """
    Custom pentagon prism block (used in AlphaBlock)
    """    
    def __init__(self, name, rgba=None):
        """
        Args:
            name (str): Name of the object
            rgba (list/tuple): 4-element (r,g,b,a) color specification
        """
        # Default color if none provided
        default_rgba = [0.5, 0.5, 0.5, 1]
        self.rgba = rgba if rgba is not None else default_rgba
    
        super().__init__(
            xml_path_completion("custom_objects/pentagon_prism_block/pentagon_prism_block.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True
        )
    
    def _get_object_subtree(self):
        # Get the base XML tree
        tree = super()._get_object_subtree()
        
        # # Find the material and update its rgba
        # print("Available materials:", [elem.attrib for elem in tree.findall(".//material")])
        
        # import xml.etree.ElementTree as ET
        # print(ET.tostring(tree, encoding='unicode'))  # Print the entire XML

        # material = tree.find(".//material[@name='pentagon_prism_mat']")
        # if material is not None:
        #     material.set("rgba", " ".join(map(str, self.rgba)))
        
        # Ensure geom uses the updated material
        geom = tree.find(".//geom[@mesh='pentagon_prism_mesh']")
        if geom is not None:
            geom.set("material", "pentagon_prism_mat")
            # Force color by adding rgba directly to geom as well
            geom.set("rgba", " ".join(map(str, self.rgba)))
        
        # Ensure the visual geom uses the updated material
        geom_visual = tree.find(".//geom[@name='g0_visual']")
        if geom_visual is not None:
            geom_visual.set("material", "pentagon_prism_mat")
            # Force color by adding rgba directly to geom as well
            geom_visual.set("rgba", " ".join(map(str, self.rgba)))
        
        return tree

class CustomStarObject(MujocoXMLObject):
    """
    Custom star prism block (used in AlphaBlock)
    """    
    def __init__(self, name, rgba=None):
        """
        Args:
            name (str): Name of the object
            rgba (list/tuple): 4-element (r,g,b,a) color specification
        """
        # Default color if none provided
        default_rgba = [0.5, 0.5, 0.5, 1]
        self.rgba = rgba if rgba is not None else default_rgba
    
        super().__init__(
            xml_path_completion("custom_objects/star_prism_block/star_prism_block.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True
        )
    
    def _get_object_subtree(self):
        # Get the base XML tree
        tree = super()._get_object_subtree()
        
        # # Find the material and update its rgba
        # print("Available materials:", [elem.attrib for elem in tree.findall(".//material")])
        
        # import xml.etree.ElementTree as ET
        # print(ET.tostring(tree, encoding='unicode'))  # Print the entire XML

        # material = tree.find(".//material[@name='star_prism_mat']")
        # if material is not None:
        #     material.set("rgba", " ".join(map(str, self.rgba)))
        
        # Ensure geom uses the updated material
        geom = tree.find(".//geom[@mesh='star_prism_mesh']")
        if geom is not None:
            geom.set("material", "star_prism_mat")
            # Force color by adding rgba directly to geom as well
            geom.set("rgba", " ".join(map(str, self.rgba)))
        
        # Ensure the visual geom uses the updated material
        geom_visual = tree.find(".//geom[@name='g0_visual']")
        if geom_visual is not None:
            geom_visual.set("material", "star_prism_mat")
            # Force color by adding rgba directly to geom as well
            geom_visual.set("rgba", " ".join(map(str, self.rgba)))
        
        return tree

class CustomMoonObject(MujocoXMLObject):
    """
    Custom moon prism block (used in AlphaBlock)
    """    
    def __init__(self, name, rgba=None):
        """
        Args:
            name (str): Name of the object
            rgba (list/tuple): 4-element (r,g,b,a) color specification
        """
        # Default color if none provided
        default_rgba = [0.5, 0.5, 0.5, 1]
        self.rgba = rgba if rgba is not None else default_rgba
    
        super().__init__(
            xml_path_completion("custom_objects/moon_prism_block/moon_prism_block.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True
        )
    
    def _get_object_subtree(self):
        # Get the base XML tree
        tree = super()._get_object_subtree()
        
        # # Find the material and update its rgba
        # print("Available materials:", [elem.attrib for elem in tree.findall(".//material")])
        
        # import xml.etree.ElementTree as ET
        # print(ET.tostring(tree, encoding='unicode'))  # Print the entire XML

        # material = tree.find(".//material[@name='moon_prism_mat']")
        # if material is not None:
        #     material.set("rgba", " ".join(map(str, self.rgba)))
        
        # Ensure geom uses the updated material
        geom = tree.find(".//geom[@mesh='moon_prism_mesh']")
        if geom is not None:
            geom.set("material", "moon_prism_mat")
            # Force color by adding rgba directly to geom as well
            geom.set("rgba", " ".join(map(str, self.rgba)))
        
        # Ensure the visual geom uses the updated material
        geom_visual = tree.find(".//geom[@name='g0_visual']")
        if geom_visual is not None:
            geom_visual.set("material", "moon_prism_mat")
            # Force color by adding rgba directly to geom as well
            geom_visual.set("rgba", " ".join(map(str, self.rgba)))
        
        return tree

class FMBObject(MujocoXMLObject):
    """
    An object extracted from the corresponding assembly XML.
    """

    def __init__(self, name, fmb_xml_file):
        self.obj_id = name.split("obj")[-1]
        
        super().__init__(
            xml_path_completion(fmb_xml_file),
            name=name,
            joints=[dict(type="free", damping="0.001")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    def _get_object_subtree(self):
        tree = super()._get_object_subtree()

        #     visual_rgba = None

        #     # Find original visual color
        #     for geom in tree.findall(".//geom"):
        #         if geom.get("material"):
        #             mat = self.asset.find(f"material[@name='{geom.get('material')}']")
        #             if mat is not None:
        #                 visual_rgba = mat.get("rgba")

        #     # Apply to generated visual geoms
        #     if visual_rgba:
        #         for geom in tree.findall(".//geom"):
        #             if geom.get("name").endswith("_visual"):
        #                 geom.set("rgba", visual_rgba)

        # return tree

        # Ensure geom uses the updated material
        visual_rgba = None

        for geom in tree.findall(".//geom"):
            if geom.get("group") == "1":  # visual group, not collision
                material_name = geom.get("material")

                if material_name:
                    mat = self.asset.find(f"material[@name='{material_name}']")

                    if mat is not None:
                        rgba = mat.get("rgba")

                        if rgba:
                            geom.set("rgba", rgba)

                            # Save color for duplicated visual geom
                            visual_rgba = rgba


        # Ensure duplicated visual geom uses the same color
        if visual_rgba is not None:
            for geom in tree.findall(".//geom"):
                if geom.get("name") == "collision_visual":
                    geom.set("rgba", visual_rgba)
        return tree
