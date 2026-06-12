using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Free-fly camera. Uses the legacy Input API, which works because the project
    // is set to Active Input Handling = Both.
    public class MarlFreeCamera : MonoBehaviour
    {
        public float moveSpeed = 16.0f;
        public float fastMultiplier = 2.5f;
        public float lookSensitivity = 2.0f;
        public float scrollSpeed = 10.0f;

        private float yaw;
        private float pitch;

        private void Start()
        {
            Vector3 r = transform.eulerAngles;
            yaw = r.y;
            pitch = r.x;
        }

        private void Update()
        {
            if (Input.GetMouseButton(1))
            {
                yaw += Input.GetAxis("Mouse X") * lookSensitivity;
                pitch -= Input.GetAxis("Mouse Y") * lookSensitivity;
                pitch = Mathf.Clamp(pitch, -85f, 85f);
                transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
            }

            float speed = Input.GetKey(KeyCode.LeftShift) ? moveSpeed * fastMultiplier : moveSpeed;

            Vector3 dir = Vector3.zero;
            dir += transform.forward * Input.GetAxisRaw("Vertical");
            dir += transform.right * Input.GetAxisRaw("Horizontal");
            if (Input.GetKey(KeyCode.E)) dir += Vector3.up;
            if (Input.GetKey(KeyCode.Q)) dir += Vector3.down;
            if (dir.sqrMagnitude > 1f) dir.Normalize();

            float scroll = Input.mouseScrollDelta.y;
            transform.position += dir * speed * Time.deltaTime;
            transform.position += transform.forward * scroll * scrollSpeed * Time.deltaTime;
        }
    }
}
