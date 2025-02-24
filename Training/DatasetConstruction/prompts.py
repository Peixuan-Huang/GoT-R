REWRITE_ONE_HOP_TRIPLET_TO_QUESTION_WITHOUT_ANSWER_MULTI_QUESTION_PROMPT = '''Given a triplet of the form [head entity, relation, tail entity], rewrite the triplet as a question where the head entity is the subject entity in the question and the tail entity is the answer to the question. Note that tail entity cannot appear in this question. Give three different statements of the question as a list.
    <triplet>: ['J.K. Rowling', 'book_write', 'Harry Potter and the Philosopher's Stone']
    <question>: ['Which books were written by J.K. Rowling?', 'Which novel did J.K. Rowling author?', 'What is one of the books authored by J.K. Rowling?']
    
    <triplet>: ['Great Wall of China', 'constructed_by', 'Chinese Dynasties']
    <question>: ['Who constructed the Great Wall of China?', 'Who was responsible for the construction of the Great Wall of China?', 'Who were the builders of the Great Wall of China?']

    <triplet>: ['Eiffel Tower', 'designed_by', 'Gustave Eiffel']
    <question>: ['Who designed the Eiffel Tower?', 'Who was the designer of the Eiffel Tower?', 'Who created the design for the Eiffel Tower?']

    <triplet>: ['Mona Lisa', 'painted_by', 'Leonardo da Vinci']
    <question>: ['Who painted the Mona Lisa?', 'Which artist is famous for painting the Mona Lisa?', 'Who is the artist of the Mona Lisa?']

    <triplet>: ['Theory of Relativity', 'proposed_by', 'Albert Einstein']
    <question>: ['Who proposed the Theory of Relativity?', 'Who introduced the Theory of Relativity?', 'Who formulated the Theory of Relativity?']

    <triplet>: ['Microsoft', 'founded_by', 'Bill Gates']
    <question>: ['Who founded Microsoft?', 'Who is the founder of Microsoft?', 'Which entrepreneur is behind the founding of Microsoft?']

    <triplet>: ['Pyramids of Giza', 'built_by', 'Ancient Egyptians']
    <question>: ['Who built the Pyramids of Giza?', 'Who constructed the Pyramids of Giza?', 'Which people built the Pyramids of Giza?']
    
    <triplet>: %s
    <question>: '''
    


REWRITE_ONE_HOP_TRIPLET_TO_QUESTION_WITHOUT_ANSWER_MULTI_QUESTION_5_CHINESE_PROMPT = '''给定一个形式为[头部实体，关系，尾部实体]的三元组，将该三元组重写为一个问题，其中头部实体是问题中的主语实体，尾部实体是问题的答案。注意，尾实体不能出现在这个问题中。用中文列出问题的五个不同陈述。
<triplet>: ['J.K.罗琳', '撰写', '哈利波特与魔法石']
<question>: ['哪些书是J.K.罗琳写的?', 'J.K.罗琳写了哪本小说?', 'J.K.罗琳写的哪本书?', 'J.K.罗琳写的那本书叫什么名字?', 'J.K.罗琳写了哪本畅销书?']

<triplet>: ['中国的长城', '建造_由', '中国']
<question>: ['谁建造了中国的长城?', '谁负责修建中国的长城?', '谁是中国长城的建造者?', '中国的长城是哪个文明建造的?', '中国的长城是由哪个团体建造的?']

<triplet>: ['埃菲尔铁塔', '设计_由', '古斯塔夫•埃菲尔']
<question>: ['谁设计了埃菲尔铁塔?', '埃菲尔铁塔的设计师是谁?', '埃菲尔铁塔的设计是谁设计的?', '埃菲尔铁塔是谁设计的?', '埃菲尔铁塔背后的建筑师是谁?']

<triplet>: ['蒙娜丽莎', '绘画_由', '达·芬奇']
<question>: ['谁画的蒙娜丽莎?', '哪个画家以画《蒙娜丽莎》而闻名?', '蒙娜丽莎的作者是谁?', '蒙娜丽莎这幅画是谁画的?', '《蒙娜丽莎》是哪个画家画的?']

<triplet>: ['相对论', ' 提出_由', '阿尔伯特·爱因斯坦']
<question>: ['谁提出了相对论?', '谁提出了相对论?', '谁提出了相对论?', '谁提出了相对论的概念?', '相对论是哪位科学家提出的?']

<triplet>: ['微软', '创立_由, '比尔盖茨']
<question>: ['谁创立了微软?', '谁是微软的创始人?', '哪位企业家创立了微软?', '谁创立了微软?', '微软是由哪位个人创立的?']

<triplet>: ['吉萨金字塔','建造_由', '古埃及人']
<question>: ['谁建造了吉萨金字塔?', '谁建造了吉萨金字塔?', '吉萨金字塔是谁建造的?', '吉萨金字塔是由哪些人建造的?', '吉萨金字塔的建造者是谁?']

<triplet>: %s
<question>: '''

REWRITE_TWO_HOP_TRIPLET_TO_QUESTION_WITHOUT_ANSWER_MULTI_QUESTION_PROMPT = '''Given a triplet of the form [head entity, relation 1, temporary entity, relation 2, tail entity], rewrite the triplet as a question where the head entity is the subject entity in the question and the tail entity is the answer to the question. Note that tail entity cannot appear in this question. Give three different statements of the question as a list.
    <triplet>: ['The Eiffel Tower', 'designed_by', 'm.0d47', 'architect_architectural_structure', 'Gustave Eiffel']
    <question>: ['Who designed the Eiffel Tower?', 'Which architect was responsible for designing the Eiffel Tower?', 'Who created the design for the Eiffel Tower?']

    <triplet>: ['Theory of Evolution', 'proposed_by', 'm.0lkn', 'scientist_theory', 'Charles Darwin']
    <question>: ['Who proposed the Theory of Evolution?', 'Which scientist is credited with developing the Theory of Evolution?', 'Who introduced the Theory of Evolution?']

    <triplet>: ['Microsoft', 'founded_by', 'm.05g4b', 'entrepreneur_company', 'Bill Gates']
    <question>: ['Who founded Microsoft?', 'Which entrepreneur is known for founding Microsoft?', 'Who established Microsoft?']

    <triplet>: ['Mona Lisa', 'painted_by', 'm.0ll1', 'artist_painting', 'Leonardo da Vinci']
    <question>: ['Who painted the Mona Lisa?', 'Which artist is famous for creating the Mona Lisa?', 'Who is the painter behind the Mona Lisa?']

    <triplet>: ['The Great Wall of China', 'constructed_by', 'm.0lq3', 'civilization_structure', 'Chinese Dynasties']
    <question>: ['Who constructed the Great Wall of China?', 'Which civilization built the Great Wall of China?', 'Who was responsible for building the Great Wall of China?']

    <triplet>: %s
    <question>: '''

