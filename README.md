# Kitchenware Care: Dishwasher Compatibility Checker App

## Background
We have become comfortable using some common household appliances, but we often forget to consider their limitations. One such appliance is the dishwasher, which was invented by Joel Houghton in 1850. A dishwasher is a machine used to automatically clean dishware, cookware, and cutlery. Unlike manual dishwashing, which relies on physical scrubbing to remove soiling, the mechanical dishwasher cleans by spraying hot water, typically between 45 and 75 °C (110 and 170 °F), at the dishes, with lower temperatures of water used for delicate items. [Source: https://en.wikipedia.org/wiki/Dishwasher]

Currently, this convenient and labor-saving machine is designed to be water-efficient and energy-efficient, as it uses less energy to heat the water for washing compared to manually heating water for hand-washing. However, it's important to note that some dishes and cookware may not be dishwasher-safe, and delicate items should still be washed by hand to prevent damage. Often, we forget the care instructions we read for cookware over time or neglect to read them in the first place.

Materials to steer clear of when using your dishwasher include fine china, clay, certain plastics, wood, nonstick, and non-enameled cast iron. Materials that could get harmed by the high temperatures of dishwashers, especially on a hot wash cycle when temperatures can reach 75 °C, include aluminum, brass, and copper items, which will discolor, and lightweight aluminum containers that may mark other items they come into contact with. Nonstick pan coatings will deteriorate, and glossy, gold-colored, and hand-painted items may become dulled or fade. Fragile items and sharp-edged items may become dulled or damaged from colliding with other items or experiencing thermal stress. Glued items, such as hollow-handle knives or wooden cutting boards, may melt or soften in a dishwasher due to high temperatures and moisture, leading to wood damage.

While some may be discouraged from using dishwashers after realizing the effort required to remember which materials are not dishwasher-safe and feeling confused about how to recognize the material of the cookware they own, this project aims to build an application that will help users identify whether a kitchenware item is dishwasher-safe or not.

## Dataset
The images for the dataset were collected from Kaggle and Google, along with some self-clicked photos of Kitchenware. As a part of preprocessing, the images are cropped to square and resized to 256x256 image using the code written in [Crop_and_Resize.py](../blob/main/Crop_and_Resize.py).

Further, the images were arranged in folders based on the type of kitchen with the directory structure as follows:
```
dataset
|--dishwasher-safe
|  |-- spoon
|  |   |--img234.png
|  |   |--img65.png
|  |   |--..
|  |   ..
|  |-- fork
|  |-- plate
|  |-- ..
|  |-- ..
|  ..
 --not-dishwasher-safe
   |-- knife
   |-- wooden_chopping_board
   |-- wooden_spoon
   |-- ..
   |-- ..
   ..
```
After performing [data anylysis](../blob/main/Data_Exploration.ipynb) the following distributions of classes and sub-classes were found

![alt text](https://github.com/anushreedas/Dishwasher-safe_or_Not/blob/main/readme_images/class_dist.png "Class Distribution")

![alt text](https://github.com/anushreedas/Dishwasher-safe_or_Not/blob/main/readme_images/sub_class_dist.png "Sub-Class Distribution")

The [notebook](../blob/main/Data_Exploration.ipynb) further explores the clusters of similar images by applying KMeans Clustering on features extracted by VGG16 model and reduced by PCA algorithm provided by the Sklearn library.
