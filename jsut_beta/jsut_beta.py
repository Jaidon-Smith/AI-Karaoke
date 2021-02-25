"""jsut_beta dataset."""

import tensorflow_datasets as tfds
import os
import tensorflow as tf

# TODO(jsut_beta): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(jsut_beta): BibTeX citation
_CITATION = """
"""


class JsutBeta(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for jsut_beta dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Place the `jsut_ver1.1.zip`
  file in the `manual_dir/`.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "id": tf.string,
            "speech": tfds.features.Audio(sample_rate=48000),
            "text": tfds.features.Text(),
        }),
        supervised_keys=("text_normalized", "speech"),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
        metadata=tfds.core.MetadataDict(sample_rate=48000),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(jsut_beta): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`

    #raise Exception(dl_manager.manual_dir)
    try:
      print(dl_manager.manual_dir)
    except:
      raise Exception("manual dir not accessible")
    #archive_path = dl_manager.manual_dir / 'jsut_ver1.1.zip'
    archive_path = 'hello2/jsut_ver1.1.zip'

    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={"directory": extracted_path},
        ),
    ]

  def _generate_examples(self, directory):
    """Yields examples."""
    # TODO(jsut_beta): Yields (key, example) tuples from the dataset
    metadata_path = os.path.join(directory, 'basic5000', 'transcript_utf8.txt')
    with tf.io.gfile.GFile(metadata_path) as f:
      for line in f:
          line = line.strip()
          key, transcript = line.split(":")
          wav_path = os.path.join(directory, "basic5000", "wavs",
                                    "%s.wav" % key)
          example = {
          "id": key,
          "speech": wav_path,
          "text": transcript,
          }
          yield key, example
