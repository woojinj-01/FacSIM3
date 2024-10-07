def is_inst_name(name):

    if not isinstance(name, str):
        return False

    if len(name.split(',')) != 3:
        return False
    else:
        return True
    

def normalize_inst_name(name):

    if not is_inst_name(name):
        return None
    
    name, region, country = tuple(name.split(','))

    name_by_words = name.lower().split(' ')
    name_by_words_norm = []

    for i, word in enumerate(name_by_words):

        if i == 0:
            word_norm = word.title()

        elif word in ['and', 'at', 'of', 'in', 'by', 'the']:
            word_norm = word

        else:
            word_norm = word.title()

        name_by_words_norm.append(word_norm.strip())

    name_norm = ' '.join(name_by_words_norm)
    region_norm = region.strip().title()
    country_norm = country.strip().upper()

    return f"{name_norm}, {region_norm}, {country_norm}"


def normalize_fac_name(name):

    words = [w.strip() for w in name.split(',')]
    return ', '.join(words)


def normalize_job_name(name):

    lowercased_name = name.lower()
    capitalized_name = lowercased_name[0].upper() + lowercased_name[1:]
    return capitalized_name


def normalize_general(name):

    lowercased_name = name.lower()
    capitalized_name = lowercased_name[0].upper() + lowercased_name[1:]
    return capitalized_name