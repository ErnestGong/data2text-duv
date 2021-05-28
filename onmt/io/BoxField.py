from collections import Counter, OrderedDict
import six
import torch
import torchtext.data
import torchtext.vocab

from torchtext.data.field import RawField
from torchtext.data.field import Field

from torchtext.data.dataset import Dataset
from torchtext.data.pipeline import Pipeline
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

# class BoxCharField(Field):
#     """Pad a batch of examples using this field.

#     Pads to self.fix_length if provided, otherwise pads to the length of
#     the longest example in the batch. Prepends self.init_token and appends
#     self.eos_token if those attributes are not None. Returns a tuple of the
#     padded list and a list containing lengths of each example if
#     `self.include_lengths` is `True` and `self.sequential` is `True`, else just
#     returns the padded list. If `self.sequential` is `False`, no padding is applied.

#     Args:
#         minibatch type is list, element are tuples of tuples

#     Return:
#         padded:three dimension list
#         length: two dimension list

#     """
#     def process(self, batch, device, train):
#         """ Process a list of examples to create a torch.Tensor.

#         Pad, numericalize, and postprocess a batch and create a tensor.

#         Args:
#             batch (list(object)): A list of object from a batch of examples.
#         Returns:
#             torch.autograd.Variable: Processed object given the input
#                 and custom postprocessing Pipeline.
#         """
#         padded = self.pad(batch)
#         if self.use_vocab:
#             try:
#                 self.vocab
#             except:
#                 print("hack BoxCharField")
#                 return padded
#         tensor = self.numericalize(padded, device=device, train=train)
#         return tensor

#     def pad_char(self, element_lst, max_char_len):
#         return [list(each_elem) + [self.pad_token] * max(0, max_char_len - 
#             len(each_elem)) for each_elem in element_lst]

#     def pad(self, minibatch):
#         minibatch = list(minibatch)
#         if not self.sequential:
#             return minibatch
#         if self.fix_length is None:
#             max_len = max(len(x) for x in minibatch)
#         else:
#             max_len = self.fix_length + (
#                 self.init_token, self.eos_token).count(None) - 2
#         max_char_len = max(len(tmp_c) for x in minibatch for tmp_c in x)
#         padded, lengths = [], []
#         for x in minibatch:
#             if self.pad_first:
#                 padded.append(
#                     self.pad_char([[self.pad_token]] * max(0, max_len - len(x)), max_char_len) +
#                     ([] if self.init_token is None else self.pad_char([[self.init_token]], max_char_len)) +
#                     self.pad_char(x[-max_len:] if self.truncate_first else x[:max_len], max_char_len) +
#                     ([] if self.eos_token is None else self.pad_char([[self.eos_token]], max_char_len)))
#                 lengths.append(
#                     [0] * max(0, max_len - len(x)) +
#                     ([] if self.init_token is None else [1]) + 
#                     [len(tmp_c) for tmp_c in (x[-max_len:] if self.truncate_first else x[:max_len])] + 
#                     ([] if self.eos_token is None else [1]))
#             else:
#                 padded.append(
#                     ([] if self.init_token is None else self.pad_char([[self.init_token]], max_char_len)) +
#                     self.pad_char(x[-max_len:] if self.truncate_first else x[:max_len], max_char_len) +
#                     ([] if self.eos_token is None else self.pad_char([[self.eos_token]], max_char_len)) +
#                     self.pad_char([[self.pad_token]] * max(0, max_len - len(x)), max_char_len))
#                 lengths.append(
#                     ([] if self.init_token is None else [1]) + 
#                     [len(tmp_c) for tmp_c in (x[-max_len:] if self.truncate_first else x[:max_len])] + 
#                     ([] if self.eos_token is None else [1]) + 
#                     [0] * max(0, max_len - len(x)))
#         # lengths is the length of characters
#         if self.include_lengths:
#             return (padded, lengths)
#         return padded

#     def numericalize(self, arr, device=None, train=True):
#         """Turn a batch of examples that use this field into a Variable.

#         If the field has include_lengths=True, a tensor of lengths will be
#         included in the return value.

#         Arguments:
#             arr (List[List[str]], or tuple of (List[List[str]], List[int])):
#                 List of tokenized and padded examples, or tuple of List of
#                 tokenized and padded examples and List of lengths of each
#                 example if self.include_lengths is True.
#             device (-1 or None): Device to create the Variable's Tensor on.
#                 Use -1 for CPU and None for the currently active GPU device.
#                 Default: None.
#             train (boolean): Whether the batch is for a training set.
#                 If False, the Variable will be created with volatile=True.
#                 Default: True.
#         """
#         if self.include_lengths and not isinstance(arr, tuple):
#             raise ValueError("Field has include_lengths set to True, but "
#                              "input data is not a tuple of "
#                              "(data batch, batch lengths).")
#         if isinstance(arr, tuple):
#             arr, lengths = arr
#             lengths = torch.LongTensor(lengths)

#         if self.use_vocab:
#             if self.sequential:
#                 arr = [[[self.vocab.stoi[char] for char in x] for x in ex] for ex in arr]
#             else:
#                 raise ValueError("non sequential char field is not supported")
#                 arr = [self.vocab.stoi[x] for x in arr]

#             if self.postprocessing is not None:
#                 arr = self.postprocessing(arr, self.vocab, train)
#         else:
#             if self.tensor_type not in self.tensor_types:
#                 raise ValueError(
#                     "Specified Field tensor_type {} can not be used with "
#                     "use_vocab=False because we do not know how to numericalize it. "
#                     "Please raise an issue at "
#                     "https://github.com/pytorch/text/issues".format(self.tensor_type))
#             numericalization_func = self.tensor_types[self.tensor_type]
#             # It doesn't make sense to explictly coerce to a numeric type if
#             # the data is sequential, since it's unclear how to coerce padding tokens
#             # to a numeric type.
#             if not self.sequential:
#                 arr = [numericalization_func(x) if isinstance(x, six.string_types)
#                        else x for x in arr]
#             if self.postprocessing is not None:
#                 arr = self.postprocessing(arr, None, train)

#         arr = self.tensor_type(arr)

#         assert len(arr.size()) == 3
#         arr = arr.view(-1, arr.size(2))
#         assert len(lengths.size()) == 2
#         lengths = lengths.view(-1)

#         if self.sequential and not self.batch_first:
#             arr.t_()
#         if device == -1:
#             if self.sequential:
#                 arr = arr.contiguous()
#         else:
#             arr = arr.cuda(device)
#             if self.include_lengths:
#                 lengths = lengths.cuda(device)
#         if self.include_lengths:
#             return Variable(arr, volatile=not train), lengths
#         return Variable(arr, volatile=not train)


class BoxField(RawField):
    """Defines a datatype together with instructions for converting to Tensor.
    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.
    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.
    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        tensor_type: The torch.Tensor class that represents a batch of examples
            of this kind of data. Default: torch.LongTensor.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list,
            the field's Vocab, and train (a bool).
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy English tokenizer is
            used. Default: str.split.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor types to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    ignore = ['dtype', 'tokenize']

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=(lambda s: s.split()), include_lengths=False,
                 batch_first=False, pad_token="<pad>", unk_token="<unk>",
                 pad_first=False, is_target=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token #if self.sequential else None
        self.pad_first = pad_first
        self.is_target = is_target

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.
        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types) and not
                isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if self.sequential and isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.
        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
                and custom postprocessing Pipeline.
        """
        padded = self.pad(batch)
        if self.use_vocab:
            try:
                self.vocab
            except:
                print("hack BoxField")
                return padded
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.
        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                counter.update(x)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (-1 or None): Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU device.
                Default: None.
            train (boolean): Whether the batch is for a training set.
                If False, the Variable will be created with volatile=True.
                Default: True.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [[self.vocab.stoi[x] for x in ex ] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        # arr = self.tensor_type(arr)
        var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:    #applies to both sequential and non-sequential
            var.t_()
        # if device == -1:
        if self.sequential:
            var = var.contiguous()
        # else:
        #     arr = arr.cuda(device)
        #     if self.include_lengths:
        #         lengths = lengths.cuda(device)
        if self.include_lengths:
            return var, lengths
        return var

# TODO
class UtilizeEntPretrainField(RawField):
    """Defines a datatype together with instructions for converting to Tensor.
    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.
    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.
    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        tensor_type: The torch.Tensor class that represents a batch of examples
            of this kind of data. Default: torch.LongTensor.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list,
            the field's Vocab, and train (a bool).
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy English tokenizer is
            used. Default: str.split.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor types to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    ignore = ['dtype', 'tokenize']

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=(lambda s: s.split()), include_lengths=False,
                 batch_first=False, pad_token="<pad>", pad_index=None, unk_token="<unk>",
                 pad_first=False, is_target=False, require_idx=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token #if self.sequential else None
        self.pad_index = pad_index
        assert pad_index is not None
        self.pad_first = pad_first
        self.is_target = is_target
        self.require_idx = require_idx

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.
        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        return x

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.
        Args:
            batch (list(tuples -> object)): A list of object from a batch of examples.
            tuples: x, pretrain indexes, not pretrain indexes
        Returns:
            torch.autograd.Variable: Processed object given the input
                and custom postprocessing Pipeline.
        """
        padded = self.pad(batch)
        assert self.use_vocab
        # if self.use_vocab:
        #     try:
        #         self.vocab
        #     except:
        #         print("hack BoxField")
        #         return padded
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.

        batch (list(tuples -> object)): A list of object from a batch of examples.
            tuples: x, pretrain_val, not pretrain val, pretrain indexes, not pretrain indexes


        """
        minibatch = list(minibatch)
        # x: list of batch: seq_len -> words
        # each_seq: list of batch: var_seq_num, seq_len
        if not self.sequential:
            assert False
            # return minibatch
        assert self.fix_length is None
        # if self.fix_length is None:
        pre_seq_num = []
        pre_seq_lens = []
        x_lens = []
        not_pre_lens = []
        

        for each_batch in minibatch:
            if self.require_idx:
                x, each_seq, ent_not_pretrain = each_batch
                not_pre_lens.append(len(ent_not_pretrain))
                x_lens.append(len(x))
            else:
                each_seq = each_batch

            pre_seq_num.append(len(each_seq))
            for tmp_seq in each_seq:
                pre_seq_lens.append(len(tmp_seq))

        max_pre_seq_num = max(pre_seq_num)
        max_pre_seq_lens = max(pre_seq_lens)
        if self.require_idx:
            max_not_pretrain_len = max(not_pre_lens)
        else:
            max_not_pretrain_len = None
        # max_len = max(x_lens)
        # else:
            # max_len = self.fix_length + (
            #     self.init_token, self.eos_token).count(None) - 2
        padded_pre, lengths = [], []
        # batch, emb_len
        padded_not_pretrain = []
        if self.require_idx:
            select_idx = []
        else:
            select_idx = None
        for bid, each_batch in enumerate(minibatch):
            if self.require_idx:
                x, each_seq, ent_not_pretrain = each_batch
                padded_not_pretrain.append(ent_not_pretrain + [self.pad_token] * max(0, max_not_pretrain_len - len(ent_not_pretrain)))
                lengths.append(len(x))
            else:
                each_seq = each_batch
            curr_idx = 0
            tmp_true_idx_dict = {}
            if self.pad_first:
                assert False
            else:
                tmp_pre_seq_res = []
                for pre_idx in range(max_pre_seq_num):
                    if pre_idx < len(each_seq):
                        tmp_pre_seq_res.append(
                            list(each_seq[pre_idx][:max_pre_seq_lens]) +
                            [self.pad_token] * max(0, max_pre_seq_lens - len(each_seq[pre_idx]))
                            )
                        for each_tmp in list(each_seq[pre_idx][:max_pre_seq_lens]):
                            if not (each_tmp == self.pad_token):
                                tmp_true_idx_dict[each_tmp] = curr_idx
                            curr_idx += 1
                        curr_idx += max(0, max_pre_seq_lens - len(each_seq[pre_idx]))
                    else:
                        tmp_pre_seq_res.append(
                            [self.pad_token] * max_pre_seq_lens
                            )
                        # tmp_true_idx_dict[self.pad_token] = curr_idx

                        curr_idx += max_pre_seq_lens
                padded_pre.append(tmp_pre_seq_res)
            # create reverse index (concatenating pretrain with not pretrain)
            if self.require_idx:
                already_pad = False
                tmp_reverse_idx = []
                for tmp_cont in x:
                    if tmp_cont not in tmp_true_idx_dict:
                        # already_pad = True
                        tmp_reverse_idx.append(curr_idx + padded_not_pretrain[bid].index(tmp_cont))
                        # tmp_reverse_idx.append(tmp_true_idx_dict[self.pad_token])
                    else:
                        # assert not already_pad
                        tmp_reverse_idx.append(tmp_true_idx_dict[tmp_cont])

                select_idx.append(tmp_reverse_idx)
            
        if self.include_lengths:
            # assert False, 'Not supported yet'
            return (padded_pre, (select_idx, padded_not_pretrain), lengths)
        return (padded_pre, (select_idx, padded_not_pretrain))

    # need to both receive and construct new vocab
    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.
        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        assert False, "Not supported yet"
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                counter.update(x)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (-1 or None): Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU device.
                Default: None.
            train (boolean): Whether the batch is for a training set.
                If False, the Variable will be created with volatile=True.
                Default: True.
        self.vocab is used for not pretrained part
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if self.include_lengths:
            arr, select_idxs, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)
        else:
            arr, select_idxs = arr

        padded_pre = arr

        if self.use_vocab:
            if self.require_idx:
                vocab_use = self.vocab['pretrain']
            else:
                vocab_use = self.vocab
            padded_pre = [[[vocab_use.stoi[elem] for elem in x] for x in ex ] for ex in padded_pre]
            if self.require_idx:
                select_idxs, tmp_not_pretrain = select_idxs
                tmp_not_pretrain = [[self.vocab['not_pretrain'].stoi[elem] for elem in ex] for ex in tmp_not_pretrain]
            else:
                select_idxs, _ = select_idxs
                tmp_not_pretrain = None

            # if self.postprocessing is not None:
            #     arr = self.postprocessing(arr, self.vocab)
        else:
            assert False, "Not supported yet"
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        # arr = self.tensor_type(arr)
        # return batch first
        padded_pre = torch.tensor(padded_pre, dtype=self.dtype, device=device)

        if self.require_idx:
            # tmp_select_idx, tmp_not_pretrain = select_idxs
            select_idxs = torch.tensor(select_idxs, dtype=self.dtype, device=device)
            tmp_not_pretrain = torch.tensor(tmp_not_pretrain, dtype=self.dtype, device=device).unsqueeze(2)
            select_idxs = (select_idxs, tmp_not_pretrain)
        else:
            select_idxs = None
  
        # if not self.batch_first:    #applies to both sequential and non-sequential
        #     var.t_()
        # if device == -1:
        if self.sequential:
            padded_pre = padded_pre.contiguous()
        # else:
        #     arr = arr.cuda(device)
        #     if self.include_lengths:
        #         lengths = lengths.cuda(device)
        if self.include_lengths:
            return (padded_pre, select_idxs), lengths
        return (padded_pre, select_idxs)



class UtilizePretrainField(RawField):
    """Defines a datatype together with instructions for converting to Tensor.
    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.
    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.
    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        tensor_type: The torch.Tensor class that represents a batch of examples
            of this kind of data. Default: torch.LongTensor.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list,
            the field's Vocab, and train (a bool).
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy English tokenizer is
            used. Default: str.split.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor types to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    ignore = ['dtype', 'tokenize']

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=(lambda s: s.split()), include_lengths=False,
                 batch_first=False, pad_token="<pad>", pad_index=None, unk_token="<unk>",
                 pad_first=False, is_target=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token #if self.sequential else None
        self.pad_index = pad_index
        assert pad_index is not None
        self.pad_first = pad_first
        self.is_target = is_target

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.
        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        return x

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.
        Args:
            batch (list(tuples -> object)): A list of object from a batch of examples.
            tuples: x, pretrain indexes, not pretrain indexes
        Returns:
            torch.autograd.Variable: Processed object given the input
                and custom postprocessing Pipeline.
        """
        padded = self.pad(batch)
        assert self.use_vocab
        # if self.use_vocab:
        #     try:
        #         self.vocab
        #     except:
        #         print("hack BoxField")
        #         return padded
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.

        batch (list(tuples -> object)): A list of object from a batch of examples.
            tuples: x, pretrain_val, not pretrain val, pretrain indexes, not pretrain indexes


        """
        minibatch = list(minibatch)
        if not self.sequential:
            assert False
            # return minibatch
        assert self.fix_length is None
        # if self.fix_length is None:
        pre_seq_num = []
        pre_seq_lens = []
        not_pre_lens = []
        x_lens = []
        for x, pretrain_val, not_val, pretrain_idx, not_idx in minibatch:
            x_lens.append(len(x))
            pre_seq_num.append(len(pretrain_val))
            for tmp_seq in pretrain_val:
                pre_seq_lens.append(len(tmp_seq))
            not_pre_lens.append(len(not_val))

        max_pre_seq_num = max(pre_seq_num)
        max_pre_seq_lens = max(pre_seq_lens)
        max_not_pre_lens = max(not_pre_lens)
        max_len = max(x_lens)
        # else:
            # max_len = self.fix_length + (
            #     self.init_token, self.eos_token).count(None) - 2
        padded_pre, padded_not_pre, lengths = [], [], []
        reverse_idxs = []
        for x, pretrain_val, not_val, pretrain_idx, not_idx in minibatch:
            curr_idx = 0
            tmp_true_idx_dict = {}
            if self.pad_first:
                assert False
            else:
                tmp_pre_seq_res = []
                for pre_idx in range(max_pre_seq_num):
                    if pre_idx < len(pretrain_val):
                        tmp_pre_seq_res.append(
                            list(pretrain_val[pre_idx][:max_pre_seq_lens]) +
                            [self.pad_token] * max(0, max_pre_seq_lens - len(pretrain_val[pre_idx]))
                            )
                        for each_tmp in list(pretrain_idx[pre_idx][:max_pre_seq_lens]):
                            tmp_true_idx_dict[int(each_tmp)] = curr_idx
                            curr_idx += 1
                        curr_idx += max(0, max_pre_seq_lens - len(pretrain_val[pre_idx]))
                    else:
                        tmp_pre_seq_res.append(
                            [self.pad_token] * max_pre_seq_lens
                            )
                        curr_idx += max_pre_seq_lens
                padded_pre.append(tmp_pre_seq_res)
                padded_not_pre.append(
                    list(not_val[:max_not_pre_lens]) + 
                    [self.pad_token] * max(0, max_not_pre_lens - len(not_val))
                    )
                for tmp_each in list(not_idx[:max_not_pre_lens]):
                    tmp_true_idx_dict[int(tmp_each)] = curr_idx
                    curr_idx += 1
                curr_idx += max(0, max_not_pre_lens - len(not_val))
            # create reverse index (concatenating pretrain with not pretrain)
            already_pad = False
            tmp_reverse_idx = []
            for tmp_i in range(max_len):
                if tmp_i not in tmp_true_idx_dict:
                    already_pad = True
                    tmp_reverse_idx.append(tmp_true_idx_dict[self.pad_index])
                else:
                    assert not already_pad
                    tmp_reverse_idx.append(tmp_true_idx_dict[tmp_i])

            reverse_idxs.append(tmp_reverse_idx)
            lengths.append(len(x))
        if self.include_lengths:
            # assert False, 'Not supported yet'
            return ((padded_pre, padded_not_pre), reverse_idxs, lengths)
        return ((padded_pre, padded_not_pre), reverse_idxs)

    # need to both receive and construct new vocab
    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.
        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        assert False, "Not supported yet"
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                counter.update(x)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (-1 or None): Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU device.
                Default: None.
            train (boolean): Whether the batch is for a training set.
                If False, the Variable will be created with volatile=True.
                Default: True.
        self.vocab is used for not pretrained part
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if self.include_lengths:
            arr, reverse_idxs, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)
        else:
            arr, reverse_idxs = arr

        padded_pre, padded_not_pre = arr

        if self.use_vocab:
            padded_pre = [[[self.vocab["pretrain"].stoi[elem] for elem in x] for x in ex ] for ex in padded_pre]
            padded_not_pre = [[self.vocab["not_pretrain"].stoi[x] for x in ex ] for ex in padded_not_pre]

            # if self.postprocessing is not None:
            #     arr = self.postprocessing(arr, self.vocab)
        else:
            assert False, "Not supported yet"
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        # arr = self.tensor_type(arr)
        # return batch first
        padded_pre = torch.tensor(padded_pre, dtype=self.dtype, device=device)
        padded_not_pre = torch.tensor(padded_not_pre, dtype=self.dtype, device=device)
        reverse_idxs = torch.tensor(reverse_idxs, dtype=self.dtype, device=device)
  
        # if not self.batch_first:    #applies to both sequential and non-sequential
        #     var.t_()
        # if device == -1:
        if self.sequential:
            padded_pre = padded_pre.contiguous()
            padded_not_pre = padded_not_pre.contiguous()
        # else:
        #     arr = arr.cuda(device)
        #     if self.include_lengths:
        #         lengths = lengths.cuda(device)
        if self.include_lengths:
            return (padded_pre, padded_not_pre, reverse_idxs), lengths
        return (padded_pre, padded_not_pre, reverse_idxs)