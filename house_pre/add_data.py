from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="my_geocoder")

def get_coordinates(location, street, block=None):
    # 주소 문자열 생성 (block 정보는 선택적으로 추가)
    address = f"{location}, {street}"
    if block:
        address += f", Block {block}"

    # 좌표 정보 가져오기
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"Error geocoding address: {e}")
    return None


location = "Tampa"
street = "E Maclaurin Dr"
block = '678'

coordinates = get_coordinates(location, street, block)
if coordinates:
    latitude, longitude = coordinates
    print(f"Coordinates: {latitude}, {longitude}")
