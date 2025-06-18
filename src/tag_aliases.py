# 🎯 优雅的标签别名映射系统
TAG_ALIASES = {
    # NSFW相关标签映射
    "mature_female": ["mature woman", "adult woman", "mature lady", "成熟女性", "熟女", "older_woman", "milf"],
    "mature_male": ["mature man", "adult man", "mature gentleman", "成熟男性", "older_man", "daddy"],
    "young_adult": ["teen", "teenager", "young woman", "young man", "青年", "young_female", "young_male"],
    
    # 身体部位标签
    "large_breasts": ["big breasts", "huge breasts", "巨乳", "大胸", "huge_boobs", "big_boobs", "voluptuous"],
    "small_breasts": ["flat chest", "tiny breasts", "贫乳", "小胸", "petite_breasts", "small_boobs"],
    "thick_thighs": ["thicc thighs", "plump thighs", "粗腿", "wide_thighs", "meaty_thighs"],
    "wide_hips": ["broad hips", "curvy hips", "宽臀", "thick_hips", "curvaceous"],
    "looking_at_viewer": ["eye contact", "direct gaze", "staring", "注视观者", "looking_forward", "direct_eye_contact"],
    
    # 场景和环境
    "basement": ["underground", "cellar", "地下室", "地下", "dungeon", "underground_room"],
    "office": ["workplace", "business", "办公室", "职场", "corporate", "work_environment"],
    "bedroom": ["bed room", "sleeping room", "卧室", "睡房", "master_bedroom", "private_room"],
    "bathroom": ["bath room", "shower room", "浴室", "盥洗室", "washroom", "restroom"],
    "classroom": ["school room", "教室", "学校", "academy", "educational_setting"],
    "kitchen": ["cooking area", "厨房", "dining", "culinary_space"],
    
    # AI绘画质量标签
    "masterpiece": ["high quality", "best quality", "finest", "杰作", "高质量", "premium_quality"],
    "ultra_detailed": ["extremely detailed", "highly detailed", "超详细", "极致细节", "intricate_details"],
    "realistic": ["photorealistic", "lifelike", "真实", "写实", "photo_realistic", "life_like"],
    "anime": ["manga style", "japanese animation", "动漫", "日式动画", "anime_style", "manga"]
}

# 智能查询意图识别模式
INTENT_PATTERNS = {
    "artist": [
        "画师", "artist", "作者", "creator", "画家", "插画师", "绘师",
        "style", "风格", "who drew", "who made", "谁画的", "by_"
    ],
    "nsfw": [
        "nsfw", "成人", "色情", "性", "裸体", "nude", "sex", "adult",
        "18+", "r18", "hentai", "工口", "黄图", "mature", "explicit"
    ],
    "character": [
        "角色", "character", "人物", "girl", "boy", "女孩", "男孩",
        "waifu", "老婆", "萌妹", "美少女", "1girl", "1boy", "solo"
    ]
}