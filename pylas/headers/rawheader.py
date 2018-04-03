import ctypes
import datetime
import enum
import logging

from .. import compression
from .. import errors

logger = logging.getLogger(__name__)

LAS_FILE_SIGNATURE = b'LASF'
PROJECT_NAME = b'pylas'


class GpsTimeType(enum.IntEnum):
    WEEK_TIME = 0
    STANDARD = 1


class GlobalEncoding(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_gps_time_type', ctypes.c_uint16, 1),
        ('waveform_internal', ctypes.c_uint16, 1),  # 1.3
        ('waveform_external', ctypes.c_uint16, 1),  # 1.3
        ('synthetic_return_numbers', ctypes.c_uint16, 1),  # 1.3
        ('wkt', ctypes.c_uint16, 1),  # 1.4
        ('reserved', ctypes.c_uint16, 11),
    ]

    def are_waveform_flag_equal(self):
        return self.waveform_internal == self.waveform_external

    @property
    def gps_time_type(self):
        if self._gps_time_type:
            return GpsTimeType.STANDARD
        else:
            return GpsTimeType.WEEK_TIME

    @gps_time_type.setter
    def gps_time_type(self, value):
        self._gps_time_type = value


class RawHeader1_1(ctypes.LittleEndianStructure):
    _version_ = '1.1'
    _pack_ = 1
    _fields_ = [
        ('file_signature', ctypes.c_char * 4),
        ('file_source_id', ctypes.c_uint16),
        ('global_encoding', GlobalEncoding),
        ('guid_data_1', ctypes.c_uint32),
        ('guid_data_2', ctypes.c_uint16),
        ('guid_data_3', ctypes.c_uint16),
        ('guid_data_4', ctypes.c_uint8 * 8),
        ('version_major', ctypes.c_uint8),
        ('version_minor', ctypes.c_uint8),
        ('system_identifier', ctypes.c_char * 32),
        ('generating_software', ctypes.c_char * 32),
        ('creation_day_of_year', ctypes.c_uint16),
        ('creation_year', ctypes.c_uint16),
        ('header_size', ctypes.c_uint16),
        ('offset_to_point_data', ctypes.c_uint32),
        ('number_of_vlr', ctypes.c_uint32),
        ('_point_data_format_id', ctypes.c_uint8),
        ('point_data_record_length', ctypes.c_uint16),
        ('legacy_number_of_point_records', ctypes.c_uint32),
        ('legacy_number_of_points_by_return', ctypes.c_uint32 * 5),
        ('x_scale', ctypes.c_double),
        ('y_scale', ctypes.c_double),
        ('z_scale', ctypes.c_double),
        ('x_offset', ctypes.c_double),
        ('y_offset', ctypes.c_double),
        ('z_offset', ctypes.c_double),
        ('x_max', ctypes.c_double),
        ('x_min', ctypes.c_double),
        ('y_max', ctypes.c_double),
        ('y_min', ctypes.c_double),
        ('z_max', ctypes.c_double),
        ('z_min', ctypes.c_double),
    ]

    def __init__(self):
        super().__init__(
            file_signature=LAS_FILE_SIGNATURE,
            version=self._version_,
            generating_software=PROJECT_NAME,
            header_size=LAS_HEADERS_SIZE[self._version_],
            offset_to_point_data=LAS_HEADERS_SIZE[self._version_],
            x_scale=0.001,
            y_scale=0.001,
            z_scale=0.001
        )

    @property
    def number_of_point_records(self):
        return self.legacy_number_of_point_records

    @number_of_point_records.setter
    def number_of_point_records(self, value):
        self.legacy_number_of_point_records = value

    @property
    def number_of_points_by_return(self):
        return self.legacy_number_of_points_by_return

    @number_of_points_by_return.setter
    def number_of_points_by_return(self, value):
        if len(value) > 5:
            logger.warning('Received return numbers up to {}, truncating to 5 for header.'.format(len(value)))
        self.legacy_number_of_points_by_return = tuple(value[:5])

    @property
    def version(self):
        return "{}.{}".format(self.version_major, self.version_minor)

    @version.setter
    def version(self, new_version):
        try:
            self.header_size = LAS_HEADERS_SIZE[str(new_version)]
            self.offset_to_point_data = self.header_size
        except KeyError:
            raise errors.FileVersionNotSupported(new_version)
        self.version_major, self.version_minor = map(int, new_version.split('.'))

    @property
    def date(self):
        try:
            return datetime.date(self.creation_year, 1, 1) + datetime.timedelta(self.creation_day_of_year - 1)
        except ValueError:
            return None

    @date.setter
    def date(self, date):
        self.creation_year = date.year
        self.creation_day_of_year = date.timetuple().tm_yday

    def write_to(self, out_stream):
        out_stream.write(bytes(self))

    @property
    def point_data_format_id(self):
        return compression.compressed_id_to_uncompressed(self._point_data_format_id)

    @point_data_format_id.setter
    def point_data_format_id(self, value):
        self._point_data_format_id = value

    @property
    def are_points_compressed(self):
        return compression.is_point_format_compressed(self._point_data_format_id)

    def __repr__(self):
        return '<LasHeader({})>'.format(self.version)


class RawHeader1_2(RawHeader1_1):
    _version_ = '1.2'


class RawHeader1_3(RawHeader1_2):
    _version_ = '1.3'
    _fields_ = [
        ('start_of_waveform_data_packet_record', ctypes.c_uint64)
    ]


class RawHeader1_4(RawHeader1_3):
    _version_ = '1.4'
    _fields_ = [
        ('start_of_first_evlr', ctypes.c_uint64),
        ('number_of_evlr', ctypes.c_uint32),
        ('number_of_point_records', ctypes.c_uint64),
        ('_number_of_points_by_return', ctypes.c_uint64 * 15)
    ]

    @property
    def number_of_points_by_return(self):
        return self._number_of_points_by_return

    @number_of_points_by_return.setter
    def number_of_points_by_return(self, value):
        value = tuple(value)
        self.legacy_number_of_points_by_return = value[:5]
        if len(value) > 15:
            logger.warning('Received return numbers up to {}, truncating to 15 for header.'.format(len(value)))
        self._number_of_points_by_return = value[:15]


class HeaderFactory:
    version_to_header = {
        '1.1': RawHeader1_1,
        '1.2': RawHeader1_2,
        '1.3': RawHeader1_3,
        '1.4': RawHeader1_4
    }
    offset_to_major_version = RawHeader1_1.version_major.offset

    @classmethod
    def header_class_for_version(cls, version):
        try:
            return cls.version_to_header[str(version)]
        except KeyError:
            raise errors.FileVersionNotSupported(version)

    @classmethod
    def new(cls, version):
        return cls.header_class_for_version(version)()

    @classmethod
    def read_from_stream(cls, stream):
        version = cls.peek_file_version(stream)
        header_class = cls.header_class_for_version(version)
        return header_class.from_buffer(bytearray(stream.read(ctypes.sizeof(header_class))))

    @classmethod
    def from_mmap(cls, mmap):
        version = cls.peek_file_version(mmap)
        return cls.header_class_for_version(version).from_buffer(mmap)

    @classmethod
    def peek_file_version(cls, stream):
        old_pos = stream.tell()
        stream.seek(cls.offset_to_major_version)
        major = int.from_bytes(stream.read(ctypes.sizeof(ctypes.c_uint8)), 'little')
        minor = int.from_bytes(stream.read(ctypes.sizeof(ctypes.c_uint8)), 'little')
        stream.seek(old_pos)
        return '{}.{}'.format(major, minor)

    @classmethod
    def convert_header(cls, old_header, new_version):
        new_header_class = cls.header_class_for_version(new_version)

        b = bytearray(old_header)
        b += b'\x00' * (ctypes.sizeof(new_header_class) - len(b))
        new_header = new_header_class.from_buffer(b)
        new_header.version = new_version

        return new_header


LAS_HEADERS_SIZE = {
    '1.1': ctypes.sizeof(RawHeader1_1),
    '1.2': ctypes.sizeof(RawHeader1_1),
    '1.3': ctypes.sizeof(RawHeader1_3),
    '1.4': ctypes.sizeof(RawHeader1_4),
}
