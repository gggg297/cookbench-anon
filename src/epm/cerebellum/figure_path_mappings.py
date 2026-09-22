# figure_path_mappings.py
"""
Figure目录图片路径完整映射配置文件
基于现有figure目录结构，为local_actions.py中的高级功能提供完整的图片路径映射
"""

import os

# 基础路径配置
BASE_FIGURE_DIR = "figure"

def get_figure_path(*path_parts):
    """获取figure目录下的完整路径"""
    return os.path.join(BASE_FIGURE_DIR, *path_parts)

# ========================================
# 1. Computer目录 - 游戏界面相关图片
# ========================================
COMPUTER_PATHS = {
    # 基础界面元素
    "assessment": get_figure_path("computer", "assessment.png"),
    "clock": get_figure_path("computer", "clock.png"),
    "cross": get_figure_path("computer", "cross.png"),
    "submit": get_figure_path("computer", "submit.png"),
    "decorations": get_figure_path("computer", "decorations.png"),
    "perks": get_figure_path("computer", "perks.png"),
    "liquid": get_figure_path("computer", "liquid.png"),
    "pour": get_figure_path("computer", "pour.png"),

    # 评分相关 (对应local_actions.py的recognize_star_rating功能)
    "taste": get_figure_path("computer", "taste.png"),
    "technique": get_figure_path("computer", "technique.png"),
    "temperature": get_figure_path("computer", "temperature.png"),
    "flavor": get_figure_path("computer", "flavor.png"),
    "star": get_figure_path("computer", "star.png"),
    "overall_score": get_figure_path("computer", "overall score.png"),
    "realization_time": get_figure_path("computer", "realization time.png"),

    # 完美评分
    "flavors_perfect": get_figure_path("computer", "flavors-perfect.png"),
    "technique_perfect": get_figure_path("computer", "technique-perfect.png"),
    "temperature_perfect": get_figure_path("computer", "temperature-perfect.png"),

    # 客人反馈相关 (对应local_actions.py的get_customer_feedback功能)
    "assess_icon": get_figure_path("computer", "assess-icon.png"),
    "guest_complaints": get_figure_path("computer", "guest complaints.png"),
    "guest_complaints_alt": get_figure_path("computer", "Gest Complaints.png"),
    "complaint_content": get_figure_path("computer", "complaint content.png"),

    # 订单管理按钮
    "order_manager_button": get_figure_path("computer", "order-manager-button.png"),
}

# 装饰主题
DECORATIONS_PATHS = {
    "60s_button": get_figure_path("computer", "decorations", "60s-button.png"),
    "60s_in_use": get_figure_path("computer", "decorations", "60s-in-use.png"),
    "basic_button": get_figure_path("computer", "decorations", "basic-button.png"),
    "basic_in_use": get_figure_path("computer", "decorations", "basic-in-use.png"),
    "countryside_button": get_figure_path("computer", "decorations", "countryside-button.png"),
    "countryside_in_use": get_figure_path("computer", "decorations", "countryside-in-use.png"),
    "extravagant_button": get_figure_path("computer", "decorations", "extravagant-button.png"),
    "extravagant_in_use": get_figure_path("computer", "decorations", "extravagant-in-use.png"),
    "future_button": get_figure_path("computer", "decorations", "future-button.png"),
    "future_in_use": get_figure_path("computer", "decorations", "future-in-use.png"),
    "horror_button": get_figure_path("computer", "decorations", "horror-button.png"),
    "horror_in_use": get_figure_path("computer", "decorations", "horror-in-use.png"),
    "hotel_button": get_figure_path("computer", "decorations", "hotel-button.png"),
    "hotel_in_use": get_figure_path("computer", "decorations", "hotel-in-use.png"),
    "modern_button": get_figure_path("computer", "decorations", "modern-button.png"),
    "modern_in_use": get_figure_path("computer", "decorations", "modern-in-use.png"),
    "temple": get_figure_path("computer", "decorations", "temple.png"),
    "temple_in_use": get_figure_path("computer", "decorations", "temple-in-use.png"),
    "weave_button": get_figure_path("computer", "decorations", "weave-button.png"),
    "weave_in_use": get_figure_path("computer", "decorations", "weave-in-use.png"),
}

# 技能/特长
PERKS_PATHS = {
    "steady_hands": get_figure_path("computer", "perks", "steady hands.png"),
}

# ========================================
# 2. Order目录 - 订单和菜品相关图片 (对应local_actions.py的order_dish和scroll_and_find_ingredient功能)
# ========================================
ORDER_PATHS = {
    # 界面控制元素
    "order_button": get_figure_path("computer", "order", "order-button.png"),
    "order_top_scroll": get_figure_path("computer", "order", "order-top-scroll.png"),
    "order_bottom_scroll": get_figure_path("computer", "order", "order-bottom-scroll.png"),
    "search_button": get_figure_path("computer", "order", "search-button.png"),
    "search_frame": get_figure_path("computer", "order", "search-frame.png"),
    "chosen_search_frame": get_figure_path("computer", "order", "chosen-search-frame.png"),

    # 菜品图片 - 按字母排序
    "baked_cod": get_figure_path("computer", "order", "Baked Cod.png"),
    "baked_cod_with_greek_salad": get_figure_path("computer", "order", "Baked Cod with Greek Salad.png"),
    "baked_potatoes_with_feta": get_figure_path("computer", "order", "Baked Potatoes with Feta.png"),
    "baked_potatoes_with_feta_tomatoes": get_figure_path("computer", "order", "Baked Potatoes with Feta & Tomatoes.png"),
    "baked_shrimp_boil": get_figure_path("computer", "order", "Baked Shrimp Boil.png"),
    "baked_trout": get_figure_path("computer", "order", "Baked Trout.png"),
    "baked_trout_with_brussels_sprouts": get_figure_path("computer", "order", "Baked Trout with Roasted Brussels Sprouts.png"),
    "barbecue_tbone_steak_potatoes_corn": get_figure_path("computer", "order", "Barbecue T-bone Steak with Potatoes and a Corn on the Cob.png"),
    "barbecue_tbone_steak_potatoes_hot_corn": get_figure_path("computer", "order", "Barbecue T-bone Steak with Potatoes and a Hot Corn on the Cob.png"),
    "beef_chuck_potatoes_garlic": get_figure_path("computer", "order", "Beef Chuck with Potatoes and Garlic Sauce.png"),
    "beef_stroganoff_fusilli": get_figure_path("computer", "order", "Beef Stroganoff over Buttered Fusilli.png"),
    "blended_fusilli_aglio_olio": get_figure_path("computer", "order", "Blended Fusilli Aglio, Olio e Peperoncino.png"),
    "braaains_shake": get_figure_path("computer", "order", "Braaains Shake.png"),
    "brussels_sprouts_pancetta_chicken": get_figure_path("computer", "order", "Brussels Sprouts with Pancetta & Chicken Wings.png"),
    "brussels_sprouts_pancetta_swordfish": get_figure_path("computer", "order", "Brussels Sprouts with Pancetta & Grilled Swordfish.png"),
    "caldo_verde": get_figure_path("computer", "order", "Caldo Verde.png"),
    "chicken_leg_caprese": get_figure_path("computer", "order", "Chicken Leg with Caprese Salad.png"),
    "chicken_pumpkin_stew": get_figure_path("computer", "order", "Chicken Pumpkin Stew.png"),
    "chicken_pumpkin_stew_tomatoes": get_figure_path("computer", "order", "Chicken Pumpkin Stew with Tomatoes.png"),
    "chicken_tikka_masala": get_figure_path("computer", "order", "Chicken Tikka Masala.png"),
    "chicken_tikka_masala_potatoes": get_figure_path("computer", "order", "Chicken Tikka Masala with Potatoes.png"),
    "chicken_tortellini_soup": get_figure_path("computer", "order", "Chicken Tortellini Soup.png"),
    "chicken_tortellini_soup_croutons": get_figure_path("computer", "order", "Chicken Tortellini Soup with Croutons.png"),
    "chinese_egg_drop_soup": get_figure_path("computer", "order", "Chinese Egg Drop Soup.png"),
    "chum_bucket": get_figure_path("computer", "order", "Chum Bucket.png"),
    "chunky_gazpacho": get_figure_path("computer", "order", "Chunky Gazpacho.png"),
    "coffee_seasoned_steak_corn": get_figure_path("computer", "order", "Coffee-seasoned steak with corn on the cob.png"),
    "corn_chowder": get_figure_path("computer", "order", "Corn Chowder.png"),
    "corn_scallop_bacon_chowder": get_figure_path("computer", "order", "Corn, Scallop and Bacon Chowder.png"),
    "cowboy_steak_potato_broccoli": get_figure_path("computer", "order", "Cowboy steak with baked potato and broccoli.png"),
    "currant_glazed_pork_tenderloin": get_figure_path("computer", "order", "Currant-Glazed Pork Tenderloin with Red Cabbage and Thyme Dumplings.png"),
    "double_potato_salad_mushroom": get_figure_path("computer", "order", "Double Potato Salad with Button Mushroom Sauce.png"),
    "double_potato_salad_pesto": get_figure_path("computer", "order", "Double Potato Salad with Pesto.png"),
    "duck_breast_apples": get_figure_path("computer", "order", "Duck Breast with Apples.png"),
    "duck_breast_mushrooms": get_figure_path("computer", "order", "Duck Breast with Roasted Mushrooms.png"),
    "duck_broth": get_figure_path("computer", "order", "Duck Broth.png"),
    "duck_consomme": get_figure_path("computer", "order", "Duck Consommé.png"),
    "dunkles_marzenbier_bbq_chicken": get_figure_path("computer", "order", "Dunkles Märzenbier BBQ Chicken.png"),
    "easy_chinese_egg_drop_soup": get_figure_path("computer", "order", "Easy Chinese Egg Drop Soup.png"),

    # Fast系列菜品
    "fast_beef_stroganoff_buttered_fusilli": get_figure_path("computer", "order", "Fast Beef Stroganoff over Buttered Fusilli.png"),
    "fast_beef_stroganoff_fusilli": get_figure_path("computer", "order", "Fast Beef Stroganoff over Fusilli.png"),
    "fast_caldo_verde": get_figure_path("computer", "order", "Fast Caldo Verde.png"),
    "fast_pasta_genovese": get_figure_path("computer", "order", "Fast Pasta alla Genovese.png"),
    "fast_pumpkin_soup": get_figure_path("computer", "order", "Fast Pumpkin Soup.png"),
    "fast_ratatouille": get_figure_path("computer", "order", "Fast Ratatouille.png"),

    # 更多菜品...
    "fiesta_corn_tomatoes": get_figure_path("computer", "order", "Fiesta Corn with Tomatoes.png"),
    "fiesta_corn_tomatoes_chicken": get_figure_path("computer", "order", "Fiesta Corn with Tomatoes & Spicy Chicken.png"),
    "fresh_meat": get_figure_path("computer", "order", "Fresh meat!.png"),
    "fried_garlic_shrimp": get_figure_path("computer", "order", "Fried Garlic Shrimp.png"),
    "fried_shrimp": get_figure_path("computer", "order", "Fried Shrimp.png"),
    "fried_shrimp_boil": get_figure_path("computer", "order", "Fried Shrimp Boil.png"),
    "fruit_salad": get_figure_path("computer", "order", "Fruit salad.png"),
    "fusilli_aglio_olio": get_figure_path("computer", "order", "Fusilli Aglio, Olio e Peperoncino.png"),
    "fusilli_blended_neapolitan": get_figure_path("computer", "order", "Fusilli with Blended Neapolitan Sauce.png"),
    "fusilli_neapolitan": get_figure_path("computer", "order", "Fusilli with Neapolitan Sauce.png"),
    "gazpacho": get_figure_path("computer", "order", "Gazpacho.png"),
    "german_potato_salad": get_figure_path("computer", "order", "German Potato Salad.png"),

    # 烤制菜品
    "grilled_beef_chuck_potatoes": get_figure_path("computer", "order", "Grilled Beef Chuck with Baked Potatoes.png"),
    "grilled_buffalo_wings": get_figure_path("computer", "order", "Grilled Buffalo Wings.png"),
    "grilled_buffalo_wings_fries": get_figure_path("computer", "order", "Grilled Buffalo Wings with French Fries.png"),
    "grilled_cabbage_burger": get_figure_path("computer", "order", "Grilled cabbage burger.png"),
    "grilled_lobster_tail_lime": get_figure_path("computer", "order", "Grilled lobster tail in lime sauce.png"),
    "grilled_pork_ribs_fries": get_figure_path("computer", "order", "Grilled pork ribs with french fries.png"),
    "grilled_swordfish_provencal": get_figure_path("computer", "order", "Grilled Swordfish Provencal.png"),
    "grilled_swordfish_hot_sauce": get_figure_path("computer", "order", "Grilled Swordfish with Hot Sauce.png"),
    "grilled_tbone_steak_potatoes": get_figure_path("computer", "order", "Grilled T-bone Steak and Baked Red Potatoes.png"),
    "grilled_tuna_steak": get_figure_path("computer", "order", "Grilled Tuna Steak.png"),
    "grilled_tuna_steak_orange_salad": get_figure_path("computer", "order", "Grilled Tuna Steak with Orange Salad.png"),
    "grilled_vegetables": get_figure_path("computer", "order", "Grilled vegetables.png"),
    "grilled_white_sausage": get_figure_path("computer", "order", "Grilled white sausage.png"),

    # 汉堡和其他主食
    "halloumi_burger": get_figure_path("computer", "order", "Halloumi burger.png"),
    "hamburger": get_figure_path("computer", "order", "Hamburger.png"),
    "honey_mustard_burger_fries": get_figure_path("computer", "order", "Honey-Mustard Burger with French Fries.png"),

    # 特殊主题菜品
    "head_headless_horseman": get_figure_path("computer", "order", "Head of the Headless Horseman.png"),
    "witchs_brew": get_figure_path("computer", "order", "Witch's Brew.png"),

    # 意大利菜和其他国际菜
    "italian_home_fries": get_figure_path("computer", "order", "Italian Home Fries.png"),
    "italian_home_fries_cucumber": get_figure_path("computer", "order", "Italian Home Fries with Cucumber Salad.png"),
    "kung_pao_chicken": get_figure_path("computer", "order", "Kung Pao Chicken.png"),
    "lemon_chicken_breasts": get_figure_path("computer", "order", "Lemon Chicken Breasts.png"),
    "lemon_chicken_breasts_fruit": get_figure_path("computer", "order", "Lemon Chicken Breasts with Fruit Salad.png"),
    "lemon_tart": get_figure_path("computer", "order", "Lemon Tart.png"),
    "mango_tart": get_figure_path("computer", "order", "Mango Tart.png"),

    # 海鲜类
    "lobster_tail_white_wine": get_figure_path("computer", "order", "Lobster Tail Cooked in White Wine.png"),
    "lobster_tail_wine_asparagus": get_figure_path("computer", "order", "Lobster tail Cooked in White Wine with Grilled Asparagus.png"),
    "salmon_butter_asparagus": get_figure_path("computer", "order", "Salmon in Butter Sauce with Asparagus.png"),
    "salmon_steak_potatoes": get_figure_path("computer", "order", "Salmon Steak and Boiled Potatoes.png"),
    "salmon_steak_potatoes_tomatoes": get_figure_path("computer", "order", "Salmon Steak, Potatoes, Grilled Tomatoes.png"),
    "salmon_asparagus": get_figure_path("computer", "order", "Salmon with Asparagus.png"),
    "sweet_smoky_salmon": get_figure_path("computer", "order", "Sweet and smoky salmon.png"),

    # 更多菜品
    "mango_salsa_chicken": get_figure_path("computer", "order", "Mango Salsa Chicken.png"),
    "mango_salsa_chicken_fries": get_figure_path("computer", "order", "Mango Salsa Chicken with French Fries.png"),
    "marinated_chicken_leg_caprese": get_figure_path("computer", "order", "Marinated Chicken Leg with Caprese Salad.png"),
    "marinated_kung_pao_chicken": get_figure_path("computer", "order", "Marinated Kung Pao Chicken.png"),
    "marinated_sweet_sour_pork": get_figure_path("computer", "order", "Marinated Sweet and Sour Pork.png"),
    "melting_potatoes": get_figure_path("computer", "order", "Melting Potatoes.png"),
    "melting_potatoes_egg_provence": get_figure_path("computer", "order", "Melting Potatoes with Egg de Provence.png"),
    "orange_pork_stir_fry": get_figure_path("computer", "order", "Orange Pork Stir-Fry.png"),
    "orange_pork_stir_fry_brussels": get_figure_path("computer", "order", "Orange Pork Stir Fry with Brussels Sprouts.png"),

    # 意面类
    "pasta_genovese": get_figure_path("computer", "order", "Pasta alla Genovese.png"),
    "penne_broccoli_sauce": get_figure_path("computer", "order", "Penne in Broccoli Sauce.png"),
    "penne_broccoli_mushroom": get_figure_path("computer", "order", "Penne in Broccoli and Mushroom Sauce.png"),
    "penne_salmon_sauce": get_figure_path("computer", "order", "Penne in Salmon Sauce.png"),
    "penne_fragrant_salmon": get_figure_path("computer", "order", "Penne in Fragrant Salmon Sauce.png"),

    # 猪肉类
    "pork_chops_potatoes": get_figure_path("computer", "order", "Pork Chops with Baked Potatoes.png"),
    "pork_chops_lemon_potatoes": get_figure_path("computer", "order", "Pork Chops and Lemon Baked Potatoes.png"),
    "pork_chops_egg_lemon_potatoes": get_figure_path("computer", "order", "Pork Chops with Fried Egg and Lemon Baked Potatoes.png"),
    "pork_tenderloin_mustard": get_figure_path("computer", "order", "Pork Tenderloin in Mustard Sauce.png"),
    "pork_tenderloin_vegetables": get_figure_path("computer", "order", "Pork Tenderloin with Caramelized Vegetables.png"),
    "roasted_pork_ribs": get_figure_path("computer", "order", "Roasted pork ribs.png"),
    "sweet_sour_pork": get_figure_path("computer", "order", "Sweet and Sour Pork.png"),

    # 汤类
    "pumpkin_soup": get_figure_path("computer", "order", "Pumpkin Soup.png"),
    "pumpkin_soup_croutons": get_figure_path("computer", "order", "Pumpkin Soup with Croutons.png"),
    "ratatouille": get_figure_path("computer", "order", "Ratatouille.png"),
    "red_pepper_tomato_soup": get_figure_path("computer", "order", "Red Pepper and Tomato Soup.png"),
    "red_pepper_tomato_soup_toast": get_figure_path("computer", "order", "Red Pepper and Tomato Soup with Toast.png"),
    "tomato_soup": get_figure_path("computer", "order", "Tomato Soup.png"),
    "simple_ukrainian_borscht": get_figure_path("computer", "order", "Simple Ukrainian Borscht.png"),
    "ukrainian_borscht": get_figure_path("computer", "order", "Ukrainian Borscht.png"),
    "ukrainian_borscht_egg": get_figure_path("computer", "order", "Ukrainian Borscht with Boiled Egg.png"),

    # 圣诞和节日菜品
    "roasted_christmas_ham": get_figure_path("computer", "order", "Roasted Christmas Ham.png"),
    "roasted_christmas_ham_potatoes": get_figure_path("computer", "order", "Roasted Christmas Ham with Garlic Fried Potatoes.png"),

    # 德国啤酒菜品
    "helles_marzenbier_pork": get_figure_path("computer", "order", "Helles Märzenbier Roasted Pork.png"),
    "white_sausage_beer": get_figure_path("computer", "order", "White sausage roasted in beer.png"),

    # 早餐类
    "sausage_egg_muffin": get_figure_path("computer", "order", "Sausage & Egg Muffin.png"),
    "sausage_egg_muffin_bacon": get_figure_path("computer", "order", "Sausage & Egg Muffin with Bacon.png"),
    "shakshuka": get_figure_path("computer", "order", "Shakshuka.png"),
    "spicy_shakshuka": get_figure_path("computer", "order", "Spicy Shakshuka.png"),

    # 沙拉和轻食
    "salad_eggplant_halloumi": get_figure_path("computer", "order", "Salad with eggplant, halloumi, and tomatoes.png"),
    "shrimp_salad_bruschetta": get_figure_path("computer", "order", "Shrimp Salad with Tomato Bruschetta.png"),
    "smokey_german_potato_salad": get_figure_path("computer", "order", "Smokey German Potato Salad.png"),
    "tomato_bruschetta": get_figure_path("computer", "order", "Tomato Bruschetta.png"),

    # 其他炖菜和特色菜
    "simple_chicken_pumpkin_stew": get_figure_path("computer", "order", "Simple Chicken Pumpkin Stew.png"),
    "spicy_sparkling_pork": get_figure_path("computer", "order", "Spicy Sparkling Pork.png"),
    "spicy_sparkling_pork_salad": get_figure_path("computer", "order", "Spicy Sparkling Pork with Spring Salad.png"),
    "steak_barbecue_vegetables": get_figure_path("computer", "order", "Steak with Barbecue Sauce and Vegetables.png"),
    "steak_french_fries": get_figure_path("computer", "order", "Steak with French Fries.png"),
}

# ========================================
# 3. Object-icon目录 - 食材图标
# ========================================
OBJECT_ICON_PATHS = {
    # 香料和调料 - 粉状
    "allspice_powder": get_figure_path("object-icon", "Allspice powder-icon.png"),
    "black_pepper": get_figure_path("object-icon", "Black Pepper-icon.png"),
    "cayenne_pepper_powder": get_figure_path("object-icon", "Cayenne Pepper powder-icon.png"),
    "cinnamon_ground": get_figure_path("object-icon", "Cinnamon ground-icon.png"),
    "cloves_ground": get_figure_path("object-icon", "Cloves ground-icon.png"),
    "cumin_powder": get_figure_path("object-icon", "Cumin powder-icon.png"),
    "curry_powder": get_figure_path("object-icon", "Curry powder-icon.png"),
    "ground_coffee": get_figure_path("object-icon", "Ground coffee-icon.png"),
    "lemon_pepper": get_figure_path("object-icon", "Lemon Pepper-icon.png"),

    # 新鲜香草
    "basil_leaf_fresh": get_figure_path("object-icon", "Basil Leaf fresh-icon.png"),
    "bay_leaf_fresh": get_figure_path("object-icon", "Bay Leaf fresh-icon.png"),
    "cilantro_fresh": get_figure_path("object-icon", "Cilantro Leaves fresh-icon.png"),
    "dill_fresh": get_figure_path("object-icon", "Dill fresh-icon.png"),
    "mint_leaf": get_figure_path("object-icon", "Mint Leaf-icon.png"),
    "chives": get_figure_path("object-icon", "Chives-icon.png"),

    # 干燥香草
    "basil_dried": get_figure_path("object-icon", "Basil dried-icon.png"),
    "bay_leaf_dried": get_figure_path("object-icon", "Bay Leaf dried-icon.png"),
    "cilantro_dried": get_figure_path("object-icon", "Cilantro Leaves dried-icon.png"),
    "dill_dried": get_figure_path("object-icon", "Dill dried-icon.png"),
    "fenugreek_dried": get_figure_path("object-icon", "Fenugreek Leaves dried-icon.png"),
    "garlic_dried": get_figure_path("object-icon", "Garlic dried-icon.png"),
    "herbs_provence": get_figure_path("object-icon", "Herbs de Provence dried-icon.png"),
    "horseradish_dried": get_figure_path("object-icon", "Horseradish dried-icon.png"),
    "lovage_dried": get_figure_path("object-icon", "Lovage dried-icon.png"),
    "marjoram_dried": get_figure_path("object-icon", "Marjoram dried-icon.png"),
    "mint_dried": get_figure_path("object-icon", "Mint dried-icon.png"),

    # 肉类
    "beef_chuck": get_figure_path("object-icon", "Beef Chuck-icon.png"),
    "burger_meat": get_figure_path("object-icon", "Burger Meat-icon.png"),
    "chicken_breast": get_figure_path("object-icon", "Chicken Breast-icon.png"),
    "chicken_leg": get_figure_path("object-icon", "Chicken Leg-icon.png"),
    "chicken_wing": get_figure_path("object-icon", "Chicken Wing-icon.png"),
    "duck_breast": get_figure_path("object-icon", "Duck Breast-icon.png"),
    "ham": get_figure_path("object-icon", "Ham-icon.png"),
    "bacon": get_figure_path("object-icon", "Bacon-icon.png"),

    # 海鲜
    "anchovy": get_figure_path("object-icon", "Anchovy-icon.png"),
    "cod": get_figure_path("object-icon", "Cod-icon.png"),
    "lobster_tail": get_figure_path("object-icon", "Lobster Tail-icon.png"),
    "salmon": get_figure_path("object-icon", "Salmon-icon.png"),
    "scallop": get_figure_path("object-icon", "Scallop-icon.png"),
    "shrimp": get_figure_path("object-icon", "Shrimp-icon.png"),
    "swordfish": get_figure_path("object-icon", "Swordfish-icon.png"),
    "trout": get_figure_path("object-icon", "Trout-icon.png"),
    "tuna": get_figure_path("object-icon", "Tuna-icon.png"),

    # 蔬菜
    "asparagus": get_figure_path("object-icon", "Asparagus-icon.png"),
    "beetroot": get_figure_path("object-icon", "Beetroot-icon.png"),
    "broccoli": get_figure_path("object-icon", "Broccoli-icon.png"),
    "brussels_sprouts": get_figure_path("object-icon", "Brussels Sprouts-icon.png"),
    "button_mushroom": get_figure_path("object-icon", "Button Mushroom-icon.png"),
    "carrot": get_figure_path("object-icon", "Carrot-icon.png"),
    "cob_corn": get_figure_path("object-icon", "Cob of Corn-icon.png"),
    "cucumber": get_figure_path("object-icon", "Cucumber-icon.png"),
    "eggplant": get_figure_path("object-icon", "Eggplant-icon.png"),
    "green_bell_pepper": get_figure_path("object-icon", "Green Bell Pepper-icon.png"),
    "chili_pepper": get_figure_path("object-icon", "Chili Pepper-icon.png"),
    "jalapeno": get_figure_path("object-icon", "Jalapeno-icon.png"),
    "onion": get_figure_path("object-icon", "Onion-icon.png"),
    "red_bell_pepper": get_figure_path("object-icon", "Red Bell Pepper-icon.png"),
    "red_cabbage": get_figure_path("object-icon", "Red Cabbage-icon.png"),
    "tomato": get_figure_path("object-icon", "Tomato-icon.png"),
    "zucchini": get_figure_path("object-icon", "Zucchini-icon.png"),

    # 土豆类
    "fingerling_potato": get_figure_path("object-icon", "Fingerling Potato-icon.png"),
    "potato": get_figure_path("object-icon", "Potato-icon.png"),
    "red_potato": get_figure_path("object-icon", "Red Potato-icon.png"),
    "sweet_potato": get_figure_path("object-icon", "Sweet Potato-icon.png"),

    # 水果
    "apple": get_figure_path("object-icon", "Apple-icon.png"),
    "banana": get_figure_path("object-icon", "Banana-icon.png"),
    "honey_mango": get_figure_path("object-icon", "Honey Mango-icon.png"),
    "lemon": get_figure_path("object-icon", "Lemon-icon.png"),
    "lime": get_figure_path("object-icon", "Lime-icon.png"),
    "orange": get_figure_path("object-icon", "Orange-icon.png"),

    # 奶制品和蛋类
    "egg": get_figure_path("object-icon", "Egg-icon.png"),
    "milk": get_figure_path("object-icon", "Milk-icon.png"),
    "clarified_butter": get_figure_path("object-icon", "Clarified Butter-icon.png"),
    "coconut_milk": get_figure_path("object-icon", "Coconut Milk-icon.png"),
    "sour_cream": get_figure_path("object-icon", "Sour Cream-icon.png"),
    "yogurt": get_figure_path("object-icon", "Yogurt-icon.png"),

    # 奶酪类
    "cheddar": get_figure_path("object-icon", "Cheddar-icon.png"),
    "feta": get_figure_path("object-icon", "Feta-icon.png"),
    "goat_cheese": get_figure_path("object-icon", "Goat Cheese-icon.png"),
    "gorgonzola": get_figure_path("object-icon", "Gorgonzola-icon.png"),
    "halloumi": get_figure_path("object-icon", "Halloumi-icon.png"),
    "mozzarella": get_figure_path("object-icon", "Mozzarella-icon.png"),
    "parmesan": get_figure_path("object-icon", "Parmesan-icon.png"),

    # 面食和谷物
    "fusilli": get_figure_path("object-icon", "Fusilli-icon.png"),
    "penne": get_figure_path("object-icon", "Penne-icon.png"),
    "tortellini": get_figure_path("object-icon", "Tortellini-icon.png"),
    "bread": get_figure_path("object-icon", "Bread-icon.png"),
    "english_muffin": get_figure_path("object-icon", "English Muffin-icon.png"),
    "bottom_burger_bun": get_figure_path("object-icon", "Bottom Burger Bun-icon.png"),
    "top_burger_bun": get_figure_path("object-icon", "Top Burger Bun-icon.png"),

    # 油类和醋类
    "avocado_oil": get_figure_path("object-icon", "Avocado Oil-icon.png"),
    "olive_oil": get_figure_path("object-icon", "Olive Oil-icon.png"),
    "sunflower_oil": get_figure_path("object-icon", "Sunflower Oil-icon.png"),
    "balsamic_vinegar": get_figure_path("object-icon", "Balsamic Vinegar-icon.png"),
    "bavarian_beer_vinegar": get_figure_path("object-icon", "Bavarian Beer Vinegar-icon.png"),
    "white_wine_vinegar": get_figure_path("object-icon", "White Wine Vinegar-icon.png"),

    # 调料和酱汁
    "barbecue_sauce": get_figure_path("object-icon", "Barbecue Sauce-icon.png"),
    "dijon_mustard": get_figure_path("object-icon", "Dijon Mustard-icon.png"),
    "dusseldorf_mustard": get_figure_path("object-icon", "Dusseldorf Mustard-icon.png"),
    "hoisin_sauce": get_figure_path("object-icon", "Hoisin Sauce-icon.png"),
    "hot_sauce": get_figure_path("object-icon", "Hot Sauce-icon.png"),
    "ketchup": get_figure_path("object-icon", "Ketchup-icon.png"),
    "mustard": get_figure_path("object-icon", "Mustard-icon.png"),
    "soy_sauce": get_figure_path("object-icon", "Soy Sauce-icon.png"),
    "worcestershire": get_figure_path("object-icon", "Worcestershire Sauce-icon.png"),

    # 高汤和液体调料
    "chicken_broth": get_figure_path("object-icon", "Chicken Broth-icon.png"),
    "vegetable_broth": get_figure_path("object-icon", "Vegetable Broth-icon.png"),
    "lemon_juice": get_figure_path("object-icon", "Lemon Juice-icon.png"),
    "lime_juice": get_figure_path("object-icon", "Lime Juice-icon.png"),
    "tomato_puree": get_figure_path("object-icon", "Tomato Puree-icon.png"),

    # 糖类和甜味剂
    "brown_sugar": get_figure_path("object-icon", "Brown Sugar-icon.png"),
    "sugar": get_figure_path("object-icon", "Sugar-icon.png"),
    "honey": get_figure_path("object-icon", "Honey-icon.png"),
    "elderflower_jelly": get_figure_path("object-icon", "Elderflower Jelly-icon.png"),

    # 饮品
    "cola": get_figure_path("object-icon", "Cola-icon.png"),
    "dunkles_marzenbier": get_figure_path("object-icon", "Dunkles Marzenbier-icon.png"),
    "helles_marzenbier": get_figure_path("object-icon", "Helles Marzenbier-icon.png"),
    "white_wine": get_figure_path("object-icon", "White Wine-icon.png"),

    # 其他
    "chili_flakes": get_figure_path("object-icon", "Chili Flakes-icon.png"),
    "cinnamon_stick": get_figure_path("object-icon", "Cinnamon stick-icon.png"),
    "garlic": get_figure_path("object-icon", "Garlic-icon.png"),
    "ginger": get_figure_path("object-icon", "Ginger-icon.png"),
    "horseradish": get_figure_path("object-icon", "Horseradish-icon.png"),
    "mustard_seeds": get_figure_path("object-icon", "Mustard Seeds-icon.png"),
    "mooncake": get_figure_path("object-icon", "Mooncake-icon.png"),
}

# ========================================
# 4. Store目录 - 商店界面相关图片
# ========================================
STORE_PATHS = {
    # 商店界面控制
    "store_top_scroll": get_figure_path("store", "store-top-scroll.png"),
    "store_bottom_scroll": get_figure_path("store", "store-bottom-scroll.png"),
    "buy_button": get_figure_path("store", "buy-button.png"),
    "cancel_button": get_figure_path("store", "cancel-button.png"),
    "confirm_button": get_figure_path("store", "confirm-button.png"),
    "store_background": get_figure_path("store", "store-background.png"),

    # 产品分类
    "products_tab": get_figure_path("store", "products-tab.png"),
    "spices_tab": get_figure_path("store", "spices-tab.png"),
    "liquids_tab": get_figure_path("store", "liquids-tab.png"),
    "utensils_tab": get_figure_path("store", "utensils-tab.png"),
    "miscellaneous_tab": get_figure_path("store", "miscellaneous-tab.png"),
}

# 商店产品图片
STORE_PRODUCTS_PATHS = {}
STORE_SPICES_PATHS = {}
STORE_LIQUIDS_PATHS = {}
STORE_UTENSILS_PATHS = {}
STORE_MISCELLANEOUS_PATHS = {}

# ========================================
# 5. 针对local_actions.py高级功能的专用路径配置
# ========================================
class AdvancedFunctionPaths:
    """为local_actions.py的高级功能提供专用路径配置"""

    # 功能1: scroll_and_find_ingredient 滚动查找原材料
    @staticmethod
    def get_scroll_search_paths():
        """获取滚动搜索功能的路径"""
        return {
            "top_template": ORDER_PATHS["order_top_scroll"],
            "bottom_template": ORDER_PATHS["order_bottom_scroll"],
            # 可以根据需要添加具体原材料模板路径
        }

    # 功能2: order_dish 订菜操作
    @staticmethod
    def get_order_dish_paths():
        """获取订菜操作功能的路径"""
        return {
            "order_button_template": ORDER_PATHS["order_button"],
            # 所有菜品模板都在ORDER_PATHS中
        }

    # 功能3: recognize_star_rating 识别星级评分
    @staticmethod
    def get_star_rating_paths():
        """获取星级评分识别功能的路径"""
        return {
            "taste_label_path": COMPUTER_PATHS["taste"],
            "time_label_path": COMPUTER_PATHS["realization_time"],
            "score_label_path": COMPUTER_PATHS["overall_score"],
            "star_template_path": COMPUTER_PATHS["star"],
        }

    # 功能4: get_customer_feedback 获取客人反馈
    @staticmethod
    def get_customer_feedback_paths():
        """获取客人反馈功能的路径"""
        return {
            "flavor_template_path": COMPUTER_PATHS["flavor"],
            "technique_template_path": COMPUTER_PATHS["technique"],
            "temperature_template_path": COMPUTER_PATHS["temperature"],
            "assess_icon_path": COMPUTER_PATHS["assess_icon"],
            "guest_complaints_path": COMPUTER_PATHS["guest_complaints"],
        }

# ========================================
# 6. 路径验证和管理工具
# ========================================
def get_all_paths():
    """获取所有路径的完整字典"""
    all_paths = {}
    all_paths.update(COMPUTER_PATHS)
    all_paths.update(DECORATIONS_PATHS)
    all_paths.update(PERKS_PATHS)
    all_paths.update(ORDER_PATHS)
    all_paths.update(OBJECT_ICON_PATHS)
    all_paths.update(STORE_PATHS)
    return all_paths

def validate_all_paths():
    """验证所有路径是否存在"""
    missing_files = []
    all_paths = get_all_paths()

    for name, path in all_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    return missing_files

def find_image_by_name(search_name):
    """根据名称查找图片路径"""
    search_name = search_name.lower()
    all_paths = get_all_paths()

    matches = []
    for name, path in all_paths.items():
        if search_name in name.lower():
            matches.append((name, path))

    return matches

# ========================================
# 7. 使用示例
# ========================================
if __name__ == "__main__":
    # 验证路径
    missing = validate_all_paths()
    if missing:
        print("缺失的文件:")
        for item in missing[:10]:  # 只显示前10个
            print(f"  - {item}")
        if len(missing) > 10:
            print(f"  ... 还有 {len(missing) - 10} 个文件")
    else:
        print("所有路径验证成功!")

    # 获取高级功能路径示例
    print("\n=== 高级功能路径配置示例 ===")

    # 滚动搜索功能
    scroll_paths = AdvancedFunctionPaths.get_scroll_search_paths()
    print("滚动搜索功能路径:", scroll_paths)

    # 星级评分功能
    rating_paths = AdvancedFunctionPaths.get_star_rating_paths()
    print("星级评分功能路径:", rating_paths)

    # 客人反馈功能
    feedback_paths = AdvancedFunctionPaths.get_customer_feedback_paths()
    print("客人反馈功能路径:", feedback_paths)

    # 搜索示例
    print("\n=== 搜索示例 ===")
    tomato_matches = find_image_by_name("tomato")
    print(f"搜索'tomato'的结果: {tomato_matches}")