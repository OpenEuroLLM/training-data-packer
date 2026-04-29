import itertools


def filter_to_be_deleted(data_iterator):
    return itertools.filterfalse(lambda x: 'delete' in x and x['delete'], data_iterator)
