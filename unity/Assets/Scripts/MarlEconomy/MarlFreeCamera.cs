using UnityEngine;

namespace MarlEconomy.UnityViz
{
    public class MarlFreeCamera : MonoBehaviour
    {
        public float moveSpeed = 14.0f;
        public float fastMultiplier = 2.5f;
        public float lookSensitivity = 2.0f;
        public float scrollSpeed = 8.0f;

        private float yaw;
        private float pitch;

        private void Start()
        {
            Vector3 rotation = transform.eulerAngles;
            yaw = rotation.y;
            pitch = rotation.x;
        }

        private void Update()
        {
            if (Input.GetMouseButton(1))
            {
                yaw += Input.GetAxis("Mouse X") * lookSensitivity;
                pitch -= Input.GetAxis("Mouse Y") * lookSensitivity;
                pitch = Mathf.Clamp(pitch, -85.0f, 85.0f);
                transform.rotation = Quaternion.Euler(pitch, yaw, 0.0f);
            }

            float speed = Input.GetKey(KeyCode.LeftShift) ? moveSpeed * fastMultiplier : moveSpeed;

            Vector3 direction = Vector3.zero;
            direction += transform.forward * Input.GetAxisRaw("Vertical");
            direction += transform.right * Input.GetAxisRaw("Horizontal");

            if (Input.GetKey(KeyCode.E))
            {
                direction += Vector3.up;
            }
            if (Input.GetKey(KeyCode.Q))
            {
                direction += Vector3.down;
            }

            if (direction.sqrMagnitude > 1.0f)
            {
                direction.Normalize();
            }

            float scroll = Input.mouseScrollDelta.y;
            transform.position += direction * speed * Time.deltaTime;
            transform.position += transform.forward * scroll * scrollSpeed * Time.deltaTime;
        }
    }
}
